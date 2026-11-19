from pathlib import Path
import importlib.util,json,torch
from types import SimpleNamespace
from transformers import Qwen3Config, Qwen3ForCausalLM
import program62_run as runner
from draft_interventions import Feedback, enable_lora

def make_model():
    config = Qwen3Config(vocab_size=48, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=8, pad_token_id=0, eos_token_id=47, attn_implementation="eager")
    return Qwen3ForCausalLM(config).eval()


torch.manual_seed(123)
torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parent
source = ROOT / "vendor" / "closed_loop_core.py"
spec=importlib.util.spec_from_file_location('actual_task59_decoder',source)
decode_module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(decode_module)
assert decode_module.cpu_tests()['status']=='pass'
class Backend:
    def __init__(self):
        self.model=make_model().requires_grad_(False)
        self.device=torch.device('cpu')
        self.hidden_size=self.model.config.hidden_size
    def _ids(self, prefix):
        return torch.tensor([prefix])
    def teacher_hidden(self, prefix, continuation):
        return self.model.model(input_ids=self._ids(prefix),use_cache=False).last_hidden_state[0,-1:]
    def next(self, prefix):
        hidden=self.teacher_hidden(prefix,[0])[0].detach()
        return SimpleNamespace(token_id=int(self.model.get_output_embeddings()(hidden.unsqueeze(0)).argmax()),
                               hidden=hidden)
cfg={'lora_rank':8,'lora_alpha':16,'slot_layer':0,'window':8,'max_new':9,'eos_token_id':47}
class DraftBackend(Backend):
    def next(self, prefix, codes, position):
        assert codes is None and 1 <= position <= 8
        return int(self.model(self._ids(prefix),use_cache=False).logits[0,-1].argmax())
target=Backend()
prompt=[1,4,7,9]
oracle=[]
for _ in range(cfg['max_new']):
    token=target.next(prompt+oracle).token_id
    oracle.append(token)
    if token==cfg['eos_token_id']:break
req={'request_id':'fixture','family_id':'family-a','prompt_token_ids':prompt,'expected_target_token_ids':oracle}
results={}
for arm in (json.loads((ROOT / 'experiment.json').read_text())['arm'],):
    draft=DraftBackend()
    adapter=runner.make_adapter(arm,draft,target,cfg)
    slot=adapter if arm=='kv' else None
    if arm=='lora': enable_lora(draft.model,False)
    raw=runner.request_eval(cfg,decode_module.decode,target,draft,req)
    if arm=='lora': enable_lora(draft.model,True)
    mask=runner.request_eval(cfg,decode_module.decode,target,draft,req,slot,'mask')
    assert runner.decision_signature(raw)==runner.decision_signature(mask)
    prefix=prompt+oracle[:1]
    previous=target.next(prefix[:-1])
    current=target.next(prefix)
    row={'prefix_ids':prefix,'prefix_sha256':runner.prefix_sha(prefix),
         'draft_depth': 1, 'previous_feedback':{'stub':True},'visible_from':len(prefix)-1,'target_greedy_id':current.token_id}
    prev_func=runner.feedback_hidden
    runner.feedback_hidden=lambda source:previous.hidden
    initial=[p.detach().clone() for p in adapter.parameters()]
    try:
        opt=torch.optim.AdamW(adapter.parameters(),lr=1e-3)
        loss,norm=runner.equal_token_step([(row,current.hidden)],draft,target,adapter,slot,opt)
    finally:
        runner.feedback_hidden=prev_func
    assert loss>0 and norm>0 and any(not torch.equal(a,b) for a,b in zip(initial,adapter.parameters()))
    assert all(p.grad is None for p in target.model.parameters())
    assert all(p.grad is None for p in draft.model.parameters() if not p.requires_grad)
    mode='lora' if arm=='lora' else 'true'
    actual=runner.request_eval(cfg,decode_module.decode,target,draft,req,slot,mode)
    assert actual['output_ids']==oracle
    results[arm]={'parity':True,'adapter_updated':True,'actual_verifier_output_exact':True,'FKL':loss,'gradient_norm':norm}
result={'arms':results,'status':'pass','scope':'tiny Qwen3 CPU, original A100 decoder and both actual adapter/train/eval functions. Production BF16 and complete label acquisition not run.',
        'decoder_sha256':runner.digest_file(source),'GPU_dispatches':0}
assert [runner.token_weight(depth, "front") for depth in range(1, 9)] == list(range(8, 0, -1))
assert [runner.token_weight(depth, "equal") for depth in range(1, 9)] == [1] * 8
print(json.dumps(result, indent=2))
