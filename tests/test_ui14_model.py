"""Small CPU task-bank/config/remote-code roundtrips, without loading the 3B model."""
import io
from pathlib import Path
import sys
import tempfile
import unittest
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from eaglevl.ui_task_registry import UI_TASKS
from eaglevl.model.locany.configuration_locateanything import LocateAnythingConfig
from eaglevl.model.locany.relation_modules import RelationConditionedDetailPyramid,RelationToPBD,class_balanced_focal_loss
from ui14_common import write_json,read_json
from patch_locany_checkpoint import patch_checkpoint,validate_relation_weight_keys


def tiny_branches():
    model=nn.Module()
    model.relation_pyramid=RelationConditionedDetailPyramid(32,32,adapter_bottleneck=8,num_defect_types=14,
        task_scale_router=True,set_localizer=True,task_hard_router=True,task_experts=True,set_decoder=True,
        task_expert_rank=4,set_decoder_deep_supervision=True,reference_position_encoding=True,per_level_scale_router=True)
    model.relation_pbd=RelationToPBD(32,32,num_defect_types=14,task_experts=True,dynamic_slot=True,coordinate_bridge=True,
        separate_global_geometry=True,straight_through_hard_routing=True)
    return model


class UI14ModelTests(unittest.TestCase):
    def test_processor_constructor_roundtrip_with_real_hf_serialization(self):
        # Isolate the image processor class from optional video/LMDB imports;
        # exercise its unmodified constructor and real ProcessorMixin.to_dict.
        import ast
        import __future__
        from transformers.processing_utils import ProcessorMixin
        from transformers import CLIPImageProcessor,PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        tree=ast.parse((ROOT/"eaglevl/utils/locany/processing_locateanything.py").read_text(encoding="utf-8"))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="LocateAnythingProcessor")
        module=ast.Module(body=[cls],type_ignores=[])
        namespace={"ProcessorMixin":ProcessorMixin,"__name__":"ui14_processor_contract","__package__":"eaglevl.utils.locany"}
        exec(compile(module,"processing_locateanything.py","exec",flags=__future__.annotations.compiler_flag),namespace)
        processor_cls=namespace["LocateAnythingProcessor"]
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"[UNK]":0,"<IMG_CONTEXT>":1},unk_token="[UNK]")),unk_token="[UNK]")
        image=CLIPImageProcessor()
        first=processor_cls(image,tokenizer,ui_num_tasks=14,ui_task_registry=[t.to_dict() for t in UI_TASKS])
        saved=first.to_dict()
        self.assertEqual(saved["ui_num_tasks"],14)
        second=processor_cls(image,tokenizer,**saved)
        self.assertEqual(second.ui_task_registry[13]["task_id"],13)

    def test_all_14_task_banks_and_four_families_roundtrip(self):
        torch.manual_seed(42)
        first=tiny_branches()
        pyramid=first.relation_pyramid
        self.assertEqual(pyramid.task_scale_embedding.num_embeddings,14)
        self.assertEqual(pyramid.defect_embedding.num_embeddings,14)
        self.assertEqual(pyramid.task_set_decoder.task_embedding.num_embeddings,14)
        self.assertEqual(len(pyramid.image_gate_heads),14)
        self.assertEqual(len(pyramid.family_adapters),4)
        self.assertEqual(tuple(pyramid.evidence_queries.shape),(4,8,32))
        self.assertTrue(validate_relation_weight_keys(set(first.state_dict()),{"tc_msed_stage":"m32","ui_num_tasks":14})["valid"])
        stream=io.BytesIO(); torch.save(first.state_dict(),stream); stream.seek(0)
        second=tiny_branches(); second.load_state_dict(torch.load(stream,weights_only=True))
        for key,value in first.state_dict().items(): self.assertTrue(torch.equal(value,second.state_dict()[key]))
        features=torch.randn(1,5,32)
        output=second.relation_pyramid.task_set_decoder(features,torch.tensor([13]))
        self.assertTrue(torch.isfinite(output.slot_boxes_norm1000).all())

    def test_ui5_focal_loss_is_bitwise_unchanged_with_extended_task_priors(self):
        logits=torch.tensor([-.2,.1,.8,-.6,.3]); target=torch.tensor([0.,1.,1.,0.,1.]); task=torch.arange(5)
        old=class_balanced_focal_loss(logits,target,task,positive_counts=(742,4480,3068,3267,2125),total_counts=(17604,)*5)
        self.assertTrue(torch.equal(old,class_balanced_focal_loss(logits,target,task)))
        self.assertTrue(torch.isfinite(class_balanced_focal_loss(logits,target,torch.tensor([5,7,9,11,13]))))

    def test_patch_and_config_reload_keep_14_sources_and_processor_metadata(self):
        from safetensors.torch import save_file
        import importlib.util
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); base=root/"cpt"; checkpoint=root/"checkpoint-1000"; base.mkdir(); checkpoint.mkdir()
            write_json(base/"processor_config.json",{"processor_class":"LocateAnythingProcessor"})
            config=LocateAnythingConfig(ui_num_tasks=14,ui_task_registry=[t.to_dict() for t in UI_TASKS],
                tc_msed_stage="m32",enable_ui_relation=True,relation_gate_threshold=.5,
                box_start_token_id=100,text_config={"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3","block_size":6,"text_mask_token_id":101})
            config.save_pretrained(checkpoint)
            save_file(tiny_branches().state_dict(),str(checkpoint/"model.safetensors"))
            patch_checkpoint(base_model=base,checkpoint=checkpoint,project_root=ROOT,force=True,validate_relation_weights=True)
            reloaded=LocateAnythingConfig.from_pretrained(checkpoint)
            self.assertEqual(reloaded.ui_num_tasks,14)
            self.assertEqual([r["task_id"] for r in reloaded.ui_task_registry],list(range(14)))
            self.assertEqual(read_json(checkpoint/"processor_config.json")["ui_num_tasks"],14)
            self.assertEqual(read_json(checkpoint/"ui_task_registry.json")["tasks"][7]["task_key"],"synth_cropping")
            module_spec=importlib.util.spec_from_file_location("ui14_standalone_registry",checkpoint/"ui_task_registry.py")
            module=importlib.util.module_from_spec(module_spec); sys.modules[module_spec.name]=module; module_spec.loader.exec_module(module)
            self.assertEqual(module.get_task(13).task_key,"synth_small_margin")
            keys={k for k in tiny_branches().state_dict() if "image_gate_heads.13." not in k}
            self.assertFalse(validate_relation_weight_keys(keys,{"tc_msed_stage":"m32","ui_num_tasks":14})["valid"])


if __name__=="__main__": unittest.main()
