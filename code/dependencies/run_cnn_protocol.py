"""Reconstruct and benchmark the two-room Majdi CNN paper protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESEARCH_LEVEL = ROOT / "experiments" / "research_level"
for location in (HERE, RESEARCH_LEVEL):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from common import (  # noqa: E402
    CandidateAttention, CompletionNet, Fingerprint, PointNetRegressor,
    configure_device, pack_fingerprints, padded_ranges, pointnet_predict,
    rectangular_sources, robust_subset_prediction, seed_all,
    simulate_fingerprint, stable_seed, summarize_rows, symmetric_chamfer_scores,
    train_pointnet, weighted_knn,
)
from majdi_paper_methods import mpurge_map_localize, published_mca_localize  # noqa: E402


ROOMS = {"room_a": np.asarray([20.0, 8.0, 3.0]), "room_b": np.asarray([16.0, 12.0, 3.0])}
SPACING = {"room_a": (4.0, 2.0), "room_b": (3.2, 3.0)}
MAX_PATHS = 9
PUBLISHED_RMSE_M = {
    "room_a": {"mca_eps1": 2.58, "mca_eps2": 2.60, "cnn_classifier": 1.82, "cnn_regressor": 1.84},
    "room_b": {"mca_eps1": 3.65, "mca_eps2": 3.64, "cnn_classifier": 2.14, "cnn_regressor": 1.47},
}


def reference_grid(room_name: str) -> np.ndarray:
    room = ROOMS[room_name]
    sx, sy = SPACING[room_name]
    span_x, span_y = 4 * sx, 3 * sy
    margin_x, margin_y = (room[0] - span_x) / 2.0, (room[1] - span_y) / 2.0
    return np.asarray([[margin_x + ix * sx, margin_y + iy * sy] for iy in range(4) for ix in range(5)], dtype=np.float64)


def obstacle_planes(room: np.ndarray) -> tuple[np.ndarray, ...]:
    w, d, _ = room
    return (
        np.asarray([[0.38*w,0.12*d,0],[0.38*w,0.62*d,0],[0.38*w,0.62*d,2.4],[0.38*w,0.12*d,2.4]]),
        np.asarray([[0.18*w,0.57*d,0],[0.74*w,0.57*d,0],[0.74*w,0.57*d,2.2],[0.18*w,0.57*d,2.2]]),
        np.asarray([[0.73*w,0.43*d,0],[0.73*w,0.92*d,0],[0.73*w,0.92*d,2.6],[0.73*w,0.43*d,2.6]]),
    )


def obstruction_configs(room: np.ndarray) -> tuple[tuple[np.ndarray, ...], ...]:
    a, b, c = obstacle_planes(room)
    return (tuple(), (a,), (b,), (c,), (a,b), (a,c), (a,b,c))


def ap_positions(room_name: str) -> np.ndarray:
    room = ROOMS[room_name]
    rng = np.random.default_rng(stable_seed("cnn-ap", room_name))
    xy = rng.uniform([0.5,0.5], [room[0]-0.5,room[1]-0.5], size=(60,2))
    return np.column_stack((xy, np.full(60, 2.5)))


def fp_at(room: np.ndarray, ap: np.ndarray, xy: np.ndarray, objects, namespace: str) -> Fingerprint:
    sources = rectangular_sources(room, ap[None], maximum_order=2, include_floor_ceiling=True)
    return simulate_fingerprint(
        [xy[0],xy[1],1.2], sources, maximum_paths=MAX_PATHS,
        rng=np.random.default_rng(stable_seed(namespace,*np.round(ap,7),*np.round(xy,7))),
        snr_db=20.0, obstructions=objects,
    )


def build_room_data(room_name: str, quick: bool) -> dict:
    room, refs, aps = ROOMS[room_name], reference_grid(room_name), ap_positions(room_name)
    configs = obstruction_configs(room)
    train_fps, labels, ap_ids, config_ids = [], [], [], []
    ap_limit = 6 if quick else 60
    for ap_id in range(ap_limit):
        for config_id, objects in enumerate(configs):
            for rp, xy in enumerate(refs):
                train_fps.append(fp_at(room, aps[ap_id], xy, objects, f"cnn-train-{room_name}-{ap_id}-{config_id}-{rp}"))
                labels.append(rp); ap_ids.append(ap_id); config_ids.append(config_id)
    rng = np.random.default_rng(stable_seed("cnn-test", room_name))
    test_rows = []
    test_configs = (0,1,4,6)
    count_per_config = 40 if quick else 400
    for test_condition, config_id in enumerate(test_configs):
        chosen_aps = ((7 + 11*test_condition) % ap_limit, (41 + 7*test_condition) % ap_limit)
        if quick:
            chosen_aps = (test_condition % ap_limit, (test_condition + 3) % ap_limit)
        for index in range(count_per_config):
            ap_id = chosen_aps[index % 2]
            xy = rng.uniform([0.05,0.05], [room[0]-0.05,room[1]-0.05])
            fingerprint = fp_at(room, aps[ap_id], xy, configs[config_id], f"cnn-test-{room_name}-{config_id}-{index}")
            test_rows.append({"condition": f"obstacles_{test_condition}", "config_id": config_id, "ap_id": ap_id, "xy": xy, "fingerprint": fingerprint})
    return {
        "room":room, "references":refs, "aps":aps[:ap_limit], "configs":configs,
        "train_fps":train_fps, "labels":np.asarray(labels), "ap_ids":np.asarray(ap_ids),
        "config_ids":np.asarray(config_ids), "test_rows":test_rows,
    }


class PaperCNN(nn.Module):
    def __init__(self, head: str):
        super().__init__()
        self.head_kind = head
        self.features = nn.Sequential(
            nn.Conv2d(1,32,(2,1)), nn.ReLU(),
            nn.Conv2d(32,64,(2,1)), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,(2,1)), nn.ReLU(),
        )
        if head == "classifier":
            self.head = nn.Linear(768,20)
        elif head == "regressor":
            self.head = nn.Sequential(nn.Linear(768,20),nn.ReLU(),nn.Linear(20,2))
        else:
            raise ValueError(head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).flatten(1))


def delay_scale(data: dict, train_indices: np.ndarray) -> tuple[float,float]:
    values = np.concatenate([data["train_fps"][int(index)].ranges_m for index in train_indices])
    return float(np.min(values)), float(np.max(values))


def encode_cnn(fingerprints, aps, references, room, minimum, maximum, *, layout: str = "column_major") -> np.ndarray:
    rows = []
    ref_xyz = np.column_stack((references[:,0]/room[0], references[:,1]/room[1], np.full(len(references),1.2/room[2]))).reshape(-1)
    for fingerprint, ap in zip(fingerprints,aps,strict=True):
        delays = np.zeros(9,dtype=np.float32)
        count=min(9,len(fingerprint)); delays[:count]=(fingerprint.ranges_m[:count]-minimum)/max(maximum-minimum,1e-6)
        ap_norm=np.asarray([ap[0]/room[0],ap[1]/room[1],ap[2]/room[2]])
        vector=np.concatenate((delays,ref_xyz,ap_norm))
        if layout=="column_major":
            # A 2x1 kernel acts along the six-row axis. Column-major packing is
            # therefore the only natural interpretation that places successive
            # delays next to one another in the convolution direction.
            rows.append(vector.reshape(1,6,12,order='F'))
        elif layout=="row_major":
            rows.append(vector.reshape(1,6,12))
        else:raise ValueError(layout)
    return np.asarray(rows,dtype=np.float32)


def stratified_split(data: dict) -> tuple[np.ndarray,np.ndarray]:
    rng=np.random.default_rng(stable_seed("cnn-split",float(data["room"][0]),float(data["room"][1])))
    train=[]; validation=[]
    groups=data["ap_ids"]*7+data["config_ids"]
    for group in np.unique(groups):
        ids=np.flatnonzero(groups==group); ids=ids[rng.permutation(len(ids))]
        validation.extend(ids[:4]); train.extend(ids[4:])
    return np.asarray(train),np.asarray(validation)


def train_paper_cnn(model, x, labels, targets_norm, train_ids, validation_ids, device, epochs, seed):
    seed_all(seed); model.to(device); optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    rng=np.random.default_rng(seed); history=[]
    for epoch in range(epochs):
        order=train_ids[rng.permutation(len(train_ids))]; model.train(); losses=[]
        for start in range(0,len(order),10):
            index=order[start:start+10]; xb=torch.as_tensor(x[index],device=device)
            output=model(xb)
            if model.head_kind=="classifier":
                loss=F.cross_entropy(output,torch.as_tensor(labels[index],device=device))
            else:
                loss=F.mse_loss(output,torch.as_tensor(targets_norm[index],device=device))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            vo=model(torch.as_tensor(x[validation_ids],device=device))
            if model.head_kind=="classifier": vl=F.cross_entropy(vo,torch.as_tensor(labels[validation_ids],device=device))
            else: vl=F.mse_loss(vo,torch.as_tensor(targets_norm[validation_ids],device=device))
        history.append({"epoch":epoch+1,"train_loss":float(np.mean(losses)),"validation_loss":float(vl)})
    return history


def train_attention_varying(model, values, masks, targets, labels, ap_ids, config_ids, reference_values, reference_masks, references, room, device, epochs, seed):
    seed_all(seed); model.to(device); optimizer=torch.optim.Adam(model.parameters(),lr=1.2e-3,weight_decay=1e-5)
    rng=np.random.default_rng(seed); context=np.column_stack((ap_ids/59.0,config_ids/6.0)).astype(np.float32)
    history=[]; rv=torch.as_tensor(reference_values,device=device); rm=torch.as_tensor(reference_masks,device=device); rp=torch.as_tensor(references.astype(np.float32),device=device)
    for _ in range(epochs):
        order=rng.permutation(len(values)); losses=[]; model.train()
        for start in range(0,len(order),128):
            ids=order[start:start+128]; a=torch.as_tensor(ap_ids[ids],device=device); cids=torch.as_tensor(config_ids[ids],device=device)
            pred,logits=model(torch.as_tensor(values[ids],device=device),torch.as_tensor(masks[ids],device=device),rv[a,cids],rm[a,cids],rp[None].expand(len(ids),-1,-1),torch.as_tensor(context[ids],device=device))
            loss=F.smooth_l1_loss(pred,torch.as_tensor(targets[ids],device=device))+0.15*F.cross_entropy(logits,torch.as_tensor(labels[ids],device=device))
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    return history


def train_completion(model, inputs, masks, target_ranges, room, device, epochs, seed):
    seed_all(seed);model.to(device);optimizer=torch.optim.Adam(model.parameters(),lr=1.2e-3);rng=np.random.default_rng(seed)
    target_mask=np.isfinite(target_ranges);target=np.nan_to_num(target_ranges/float(np.linalg.norm(room)),nan=0).astype(np.float32);counts=target_mask.sum(1)
    history=[]
    for _ in range(epochs):
        order=rng.permutation(len(inputs));losses=[]
        for start in range(0,len(order),256):
            ids=order[start:start+256];x=torch.as_tensor(inputs[ids],device=device);m=torch.as_tensor(masks[ids],device=device);y=torch.as_tensor(target[ids],device=device);ym=torch.as_tensor(target_mask[ids],device=device);co=torch.as_tensor(counts[ids],device=device)
            latent=torch.randn(len(ids),4,model.latent_dim,device=device);gen,logits=model(x,m,latent);vl=(F.smooth_l1_loss(gen,y[:,None].expand_as(gen),reduction='none')*ym[:,None]).sum(2)/ym.sum(1,keepdim=True).clamp_min(1);loss=vl.min(1).values.mean()+.15*F.cross_entropy(logits.mean(1),co)
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    return history


@torch.no_grad()
def predict_attention_varying(model, values, masks, test_rows, reference_values, reference_masks, references, device):
    model.eval();output=[];rv=torch.as_tensor(reference_values,device=device);rm=torch.as_tensor(reference_masks,device=device);rp=torch.as_tensor(references.astype(np.float32),device=device)
    for start in range(0,len(values),128):
        stop=min(start+128,len(values));rows=test_rows[start:stop];a=np.asarray([r['ap_id'] for r in rows]);c=np.asarray([r['config_id'] for r in rows]);context=np.column_stack((a/59.0,c/6.0)).astype(np.float32)
        pred,_=model(torch.as_tensor(values[start:stop],device=device),torch.as_tensor(masks[start:stop],device=device),rv[torch.as_tensor(a,device=device),torch.as_tensor(c,device=device)],rm[torch.as_tensor(a,device=device),torch.as_tensor(c,device=device)],rp[None].expand(stop-start,-1,-1),torch.as_tensor(context,device=device));output.append(pred.cpu().numpy())
    return np.concatenate(output)


@torch.no_grad()
def predict_completion(model, values, masks, test_rows, reference_ranges, references, room, device):
    model.eval();output=[];generator=torch.Generator(device=device).manual_seed(stable_seed('cnn-xm-test',float(room[0])))
    for start in range(0,len(values),256):
        stop=min(start+256,len(values));gen,logits=model(torch.as_tensor(values[start:stop],device=device),torch.as_tensor(masks[start:stop],device=device),torch.randn(stop-start,4,model.latent_dim,generator=generator,device=device));gen=gen.cpu().numpy()*float(np.linalg.norm(room));counts=torch.argmax(logits,2).clamp_min(1).cpu().numpy()
        for local,row in enumerate(test_rows[start:stop]):
            scores=[]
            for mode in range(4):scores.append(symmetric_chamfer_scores(gen[local,mode,:counts[local,mode]],reference_ranges[row['ap_id'],row['config_id']]))
            output.append(weighted_knn(np.mean(scores,axis=0),references,3))
    return np.asarray(output)


def run_room(room_name: str, device: torch.device, quick: bool) -> tuple[list[dict],dict]:
    data=build_room_data(room_name,quick);room,refs,aps=data['room'],data['references'],data['aps'];train_ids,val_ids=stratified_split(data);minimum,maximum=delay_scale(data,train_ids)
    sample_aps=np.asarray([aps[i] for i in data['ap_ids']]);x=encode_cnn(data['train_fps'],sample_aps,refs,room,minimum,maximum);targets=np.asarray([refs[i] for i in data['labels']],dtype=np.float32);target_norm=(targets/room[:2]).astype(np.float32)
    epochs=5 if quick else 100
    classifier=PaperCNN('classifier');regressor=PaperCNN('regressor')
    classifier_history=train_paper_cnn(classifier,x,data['labels'],target_norm,train_ids,val_ids,device,epochs,stable_seed(room_name,'classifier'))
    regressor_history=train_paper_cnn(regressor,x,data['labels'],target_norm,train_ids,val_ids,device,epochs,stable_seed(room_name,'regressor'))
    values,masks=pack_fingerprints(data['train_fps'],maximum_paths=9,range_scale_m=float(np.linalg.norm(room)),maximum_tx=1)
    context=np.column_stack((sample_aps[:,0]/room[0],sample_aps[:,1]/room[1],sample_aps[:,2]/room[2],data['config_ids']/6)).astype(np.float32)
    pointnet=PointNetRegressor(context_dim=4);point_history=train_pointnet(pointnet,values[train_ids],masks[train_ids],target_norm[train_ids],context=context[train_ids],device=device,epochs=5 if quick else 100,batch_size=128,seed=stable_seed(room_name,'pointnet'))
    # Candidate fingerprints are the 20 RPs for every AP/config pair.
    rv=np.zeros((len(aps),7,20,9,4),dtype=np.float32);rm=np.zeros((len(aps),7,20,9),dtype=bool);rr=np.full((len(aps),7,20,9),np.nan)
    lookup={(int(a),int(c),int(label)):i for i,(a,c,label) in enumerate(zip(data['ap_ids'],data['config_ids'],data['labels'],strict=True))}
    for a in range(len(aps)):
        for c in range(7):
            fps=[data['train_fps'][lookup[(a,c,r)]] for r in range(20)];rv[a,c],rm[a,c]=pack_fingerprints(fps,maximum_paths=9,range_scale_m=float(np.linalg.norm(room)),maximum_tx=1);rr[a,c]=padded_ranges(fps,9)
    attention=CandidateAttention(context_dim=2);attention_history=train_attention_varying(attention,values[train_ids],masks[train_ids],targets[train_ids],data['labels'][train_ids],data['ap_ids'][train_ids],data['config_ids'][train_ids],rv,rm,refs,room,device,5 if quick else 80,stable_seed(room_name,'attention'))
    # Completion training receives a deterministic 25% path mask and predicts the original strongest-nine set.
    partial=[]
    for index,fp in enumerate(data['train_fps']):
        rng=np.random.default_rng(stable_seed(room_name,'completion-mask',index));keep=rng.random(len(fp))>.25;partial.append(Fingerprint(fp.ranges_m[keep],fp.powers_db[keep],fp.tx_ids[keep]))
    pv,pm=pack_fingerprints(partial,maximum_paths=9,range_scale_m=float(np.linalg.norm(room)),maximum_tx=1);completion=CompletionNet(9);completion_history=train_completion(completion,pv[train_ids],pm[train_ids],padded_ranges(data['train_fps'],9)[train_ids],room,device,5 if quick else 80,stable_seed(room_name,'completion'))
    tests=data['test_rows'];test_fps=[row['fingerprint'] for row in tests];test_aps=np.asarray([aps[row['ap_id']] for row in tests]);test_x=encode_cnn(test_fps,test_aps,refs,room,minimum,maximum);tv,tm=pack_fingerprints(test_fps,maximum_paths=9,range_scale_m=float(np.linalg.norm(room)),maximum_tx=1);test_context=np.column_stack((test_aps[:,0]/room[0],test_aps[:,1]/room[1],test_aps[:,2]/room[2],np.asarray([row['config_id'] for row in tests])/6)).astype(np.float32)
    classifier.eval();regressor.eval()
    with torch.no_grad():
        logits=classifier(torch.as_tensor(test_x,device=device));weights=torch.softmax(logits,1).cpu().numpy();cnn_class=weights@refs
        cnn_reg=regressor(torch.as_tensor(test_x,device=device)).cpu().numpy()*room[:2]
    point=pointnet_predict(pointnet,tv,tm,context=test_context,device=device)*room[:2]
    attention_prediction=predict_attention_varying(attention,tv,tm,tests,rv,rm,refs,device)
    # Threshold is selected on held-out validation rows only.
    vv,vm=values[val_ids],masks[val_ids];vctx=context[val_ids];vpoint=pointnet_predict(pointnet,vv,vm,context=vctx,device=device)*room[:2]
    val_rows=[{'ap_id':int(data['ap_ids'][i]),'config_id':int(data['config_ids'][i])} for i in val_ids];vattention=predict_attention_varying(attention,vv,vm,val_rows,rv,rm,refs,device);truth=targets[val_ids]
    grid=[]
    for threshold in np.arange(0,5.01,.25):
        agree=np.linalg.norm(vpoint-vattention,axis=1)<=threshold;pred=np.where(agree[:,None],.5*(vpoint+vattention),vpoint);grid.append((float(np.mean(np.linalg.norm(pred-truth,axis=1))),float(threshold)))
    selected_threshold=min(grid)[1];agree=np.linalg.norm(point-attention_prediction,axis=1)<=selected_threshold;agreement=np.where(agree[:,None],.5*(point+attention_prediction),point)
    completion_prediction=predict_completion(completion,tv,tm,tests,rr,refs,room,device)
    rows=[]
    for i,row in enumerate(tests):
        ref_fps=[data['train_fps'][lookup[(row['ap_id'],row['config_id'],rp)]] for rp in range(20)];reference_ranges=rr[row['ap_id'],row['config_id']];q=row['fingerprint'];truth=row['xy']
        predictions={
            'majdi_cnn_classifier':cnn_class[i], 'majdi_cnn_regressor':cnn_reg[i],
            'majdi_mca_eps1':published_mca_localize(q.ranges_m,[fp.ranges_m for fp in ref_fps],refs,epsilon_m=1,k=1),
            'majdi_mca_eps2':published_mca_localize(q.ranges_m,[fp.ranges_m for fp in ref_fps],refs,epsilon_m=2,k=1),
            'pointnet_direct':point[i], 'challenger_candidate_cross_attention':attention_prediction[i],
            'challenger_agreement_gated':agreement[i], 'challenger_subset_consensus':robust_subset_prediction(q,reference_ranges,refs),
            'challenger_xm_k4_completion':completion_prediction[i],
        }
        predictions['challenger_corrected_mpurge_map']=mpurge_map_localize(q.ranges_m,[fp.ranges_m for fp in ref_fps],refs,p=6,alpha=.7,k=3,normalized_pattern=True,coverage_mode='penalty')[0]
        for method,prediction in predictions.items():
            rows.append({'protocol':'cnn_paper','room':room_name,'condition':row['condition'],'query':i,'method':method,'truth_xy_m':truth.tolist(),'prediction_xy_m':np.asarray(prediction).tolist(),'error_m':float(np.linalg.norm(np.asarray(prediction)-truth))})
    audit={'training_examples':len(data['train_fps']),'expected_training_examples':8400 if not quick else 840,'test_examples':len(tests),'expected_test_examples':1600 if not quick else 160,'train_split':len(train_ids),'validation_split':len(val_ids),'classifier_parameters':sum(p.numel() for p in classifier.parameters()),'regressor_parameters_computed':sum(p.numel() for p in regressor.parameters()),'paper_reported_regressor_parameters':36211,'selected_agreement_threshold_m':selected_threshold,'training_final_losses':{'classifier':classifier_history[-1],'regressor':regressor_history[-1],'pointnet':point_history[-1],'attention':attention_history[-1],'completion':completion_history[-1]}}
    return rows,audit


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--quick',action='store_true');parser.add_argument('--output',type=Path,default=ROOT/'research'/'paper_protocol_replications'/'cnn_protocol_results.json');args=parser.parse_args();started=time.perf_counter();device=configure_device();all_rows=[];audits={}
    for room_name in ROOMS:
        rows,audit=run_room(room_name,device,args.quick);all_rows.extend(rows);audits[room_name]=audit;print(f"finished {room_name}: {audit['test_examples']} test examples",flush=True)
    result={'schema':'majdi-cnn-paper-protocol-reconstruction-v1','status':'QUICK_SMOKE' if args.quick else 'FULL_RECONSTRUCTION_COMPLETE','evidence_tier':3,'claim':'paper-specified architecture/counts with disclosed replacement ray geometry; not exact paper reproduction','paper_specified':{'rooms_xyz_m':{k:v.tolist() for k,v in ROOMS.items()},'rp_grid':'5x4 (20)','ap_positions':60,'obstruction_configurations':7,'fingerprints_per_room':8400,'carrier_ghz':60,'bandwidth_ghz':2,'snr_db':20,'maximum_paths':9,'epochs':100,'batch_size':10,'optimizer':'Adam lr=1e-3','test_points_per_room':1600},'published_targets_rmse_m':PUBLISHED_RMSE_M,'challengers':['corrected_mpurge_map','subset_consensus','candidate_cross_attention','agreement_gated','xm_k4_completion'],'required_additional':'pointnet_direct','audits':audits,'summaries':summarize_rows(all_rows),'runtime_s':time.perf_counter()-started,'device':str(device),'rows':all_rows};args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({'status':result['status'],'runtime_s':result['runtime_s'],'audits':audits},indent=2))


if __name__=='__main__':main()
