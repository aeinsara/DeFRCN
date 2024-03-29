import os
from .meta_voc import register_meta_voc
from .meta_coco import register_meta_coco
from .meta_dota import register_meta_dota
from .meta_dior import register_meta_dior

from .builtin_meta import _get_builtin_metadata
from detectron2.data import DatasetCatalog, MetadataCatalog


# -------- COCO -------- #
def register_all_coco(root="datasets"):

    METASPLITS = [
        ("coco14_trainval_all", "coco/trainval2014", "cocosplit/datasplit/trainvalno5k.json"),
        ("coco14_trainval_base", "coco/trainval2014", "cocosplit/datasplit/trainvalno5k.json"),
        ("coco14_test_all", "coco/val2014", "cocosplit/datasplit/5k.json"),
        ("coco14_test_base", "coco/val2014", "cocosplit/datasplit/5k.json"),
        ("coco14_test_novel", "coco/val2014", "cocosplit/datasplit/5k.json"),
    ]
#     METASPLITS = [
#         ("coco14_trainval_all", "coco/train2014", "cocosplit/datasplit/trainvalno5k.json"),
#         ("coco14_trainval_base", "coco/train2014", "cocosplit/datasplit/trainvalno5k.json"),
#         ("coco14_test_all", "coco/val2014", "cocosplit/datasplit/5k.json"),
#         ("coco14_test_base", "coco/val2014", "cocosplit/datasplit/5k.json"),
#         ("coco14_test_novel", "coco/val2014", "cocosplit/datasplit/5k.json"),
#     ]
    for prefix in ["all", "novel"]:
        for shot in [1, 2, 3, 5, 10, 30]:
            for seed in range(10):
                name = "coco14_trainval_{}_{}shot_seed{}".format(prefix, shot, seed)
                METASPLITS.append((name, "coco/trainval2014", ""))

    for name, imgdir, annofile in METASPLITS:
        register_meta_coco(
            name,
            _get_builtin_metadata("coco_fewshot"),
            os.path.join(root, imgdir),
            os.path.join(root, annofile),
        )


# -------- PASCAL VOC -------- #
def register_all_voc(root="datasets"):

    METASPLITS = [
        ("voc_2007_trainval_base1", "VOC2007", "trainval", "base1", 1),
        ("voc_2007_trainval_base2", "VOC2007", "trainval", "base2", 2),
        ("voc_2007_trainval_base3", "VOC2007", "trainval", "base3", 3),
        ("voc_2012_trainval_base1", "VOC2012", "trainval", "base1", 1),
        ("voc_2012_trainval_base2", "VOC2012", "trainval", "base2", 2),
        ("voc_2012_trainval_base3", "VOC2012", "trainval", "base3", 3),
        ("voc_2007_trainval_all1", "VOC2007", "trainval", "base_novel_1", 1),
        ("voc_2007_trainval_all2", "VOC2007", "trainval", "base_novel_2", 2),
        ("voc_2007_trainval_all3", "VOC2007", "trainval", "base_novel_3", 3),
        ("voc_2012_trainval_all1", "VOC2012", "trainval", "base_novel_1", 1),
        ("voc_2012_trainval_all2", "VOC2012", "trainval", "base_novel_2", 2),
        ("voc_2012_trainval_all3", "VOC2012", "trainval", "base_novel_3", 3),
        ("voc_2007_test_base1", "VOC2007", "test", "base1", 1),
        ("voc_2007_test_base2", "VOC2007", "test", "base2", 2),
        ("voc_2007_test_base3", "VOC2007", "test", "base3", 3),
        ("voc_2007_test_novel1", "VOC2007", "test", "novel1", 1),
        ("voc_2007_test_novel2", "VOC2007", "test", "novel2", 2),
        ("voc_2007_test_novel3", "VOC2007", "test", "novel3", 3),
        ("voc_2007_test_all1", "VOC2007", "test", "base_novel_1", 1),
        ("voc_2007_test_all2", "VOC2007", "test", "base_novel_2", 2),
        ("voc_2007_test_all3", "VOC2007", "test", "base_novel_3", 3),
    ]
    for prefix in ["all", "novel"]:
        for sid in range(1, 4):
            for shot in [1, 2, 3, 5, 10]:
                for year in [2007, 2012]:
                    for seed in range(30):
                        seed = "_seed{}".format(seed)
                        name = "voc_{}_trainval_{}{}_{}shot{}".format(
                            year, prefix, sid, shot, seed
                        )
                        dirname = "VOC{}".format(year)
                        img_file = "{}_{}shot_split_{}_trainval".format(
                            prefix, shot, sid
                        )
                        keepclasses = (
                            "base_novel_{}".format(sid)
                            if prefix == "all"
                            else "novel{}".format(sid)
                        )
                        METASPLITS.append(
                            (name, dirname, img_file, keepclasses, sid)
                        )

    for name, dirname, split, keepclasses, sid in METASPLITS:
        year = 2007 if "2007" in name else 2012
        register_meta_voc(
            name,
            _get_builtin_metadata("voc_fewshot"),
            os.path.join(root, dirname),
            split,
            year,
            keepclasses,
            sid,
        )
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"

        
# -------- DOTA -------- #
def register_all_dota(root="/storage/aeen/fewshot/Datasets"):

    METASPLITS = [
        ("dota_trainval_all", "dota/trainval1024", "dota/dotasplit/datasplit/trainvalno5k_all.json"),
        ("dota_trainval_base", "dota/trainval1024", "dota/dotasplit/datasplit/trainvalno5k_all.json"),
        ("dota_trainval_novel", "dota/trainval1024", "dota/dotasplit/datasplit/trainvalno5k_all.json"),
        ("dota_test_all", "dota/test1024", "dota/dotasplit/datasplit/5k.json"),
        ("dota_test_base", "dota/test1024", "dota/dotasplit/datasplit/5k.json"),
        ("dota_test_novel", "dota/test1024", "dota/dotasplit/datasplit/5k.json"),
    ]
    for prefix in ["all", "novel"]:
        for shot in [1, 2, 3, 5, 10, 30]:
            for seed in range(10):
                name = "dota_trainval_{}_{}shot_seed{}".format(prefix, shot, seed)
                METASPLITS.append((name, "dota/trainval1024", ""))

    for name, imgdir, annofile in METASPLITS:
        register_meta_dota(
            name,
            _get_builtin_metadata("dota_fewshot"),
            os.path.join(root, imgdir),
            os.path.join(root, annofile),
        )

# -------- DIOR -------- #
def register_all_dior(root="/storage/aeen/fewshot/Datasets"):

    METASPLITS = [
        ("dior_trainval_all", "dior/JPEGImages-trainval", "dior/Annotations/dior_coco_annotations_trainval.json"),
        ("dior_trainval_base", "dior/JPEGImages-trainval", "dior/Annotations/dior_coco_annotations_trainval.json"),
        ("dior_trainval_novel", "dior/JPEGImages-trainval", "dior/Annotations/dior_coco_annotations_trainval.json"),
        ("dior_test_all", "dior/JPEGImages-test", "dior/Annotations/dior_coco_annotations_test.json"),
        ("dior_test_base", "dior/JPEGImages-test", "dior/Annotations/dior_coco_annotations_test.json"),
        ("dior_test_novel", "dior/JPEGImages-test", "dior/Annotations/dior_coco_annotations_test.json"),
    ]
    for prefix in ["all", "novel"]:
        for shot in [1, 2, 3, 5, 10, 30]:
            for seed in range(10):
                name = "dior_trainval_{}_{}shot_seed{}".format(prefix, shot, seed)
                METASPLITS.append((name, "dior/JPEGImages-trainval", ""))

    for name, imgdir, annofile in METASPLITS:
        register_meta_dior(
            name,
            _get_builtin_metadata("dior_fewshot"),
            os.path.join(root, imgdir),
            os.path.join(root, annofile),
        )
        
register_all_coco()
register_all_voc()
register_all_dota()
register_all_dior()