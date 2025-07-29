# LaneATT + LaueIoU

We provide modified LaneATT code leveraging LaneIoU-based matcher.

## Train & test
```
python tools/train.py configs/laneatt/culane/laneatt_culane_medium.py
python tools/test.py configs/laneatt/culane/laneatt_culane_medium.py {weight_path}
```

Evaluation results
```
Evaluation results for IoU threshold = 0.1
Eval category: test_all    , N: 34680, TP: 84765, FP:  4631, FN: 20121, Precision: 0.9482, Recall: 0.8082, F1: 0.8726
Eval category: test0_normal, N: 9621, TP: 31684, FP:   493, FN:  1093, Precision: 0.9847, Recall: 0.9667, F1: 0.9756
Eval category: test1_crowd , N: 8113, TP: 22761, FP:  1012, FN:  5242, Precision: 0.9574, Recall: 0.8128, F1: 0.8792
Eval category: test2_hlight, N:  486, TP:  1268, FP:    57, FN:   417, Precision: 0.9570, Recall: 0.7525, F1: 0.8425
Eval category: test3_shadow, N:  930, TP:  2349, FP:   103, FN:   527, Precision: 0.9580, Recall: 0.8168, F1: 0.8818
Eval category: test4_noline, N: 4067, TP:  7426, FP:   953, FN:  6595, Precision: 0.8863, Recall: 0.5296, F1: 0.6630
Eval category: test5_arrow , N:  890, TP:  2933, FP:    70, FN:   249, Precision: 0.9767, Recall: 0.9217, F1: 0.9484
Eval category: test6_curve , N:  422, TP:   826, FP:     5, FN:   486, Precision: 0.9940, Recall: 0.6296, F1: 0.7709
Eval category: test7_cross , N: 3122, TP:     0, FP:  1114, FN:     0, Precision: 0.0000, Recall: 0.0000, F1: 0.0000
Eval category: test8_night , N: 7029, TP: 15518, FP:   824, FN:  5512, Precision: 0.9496, Recall: 0.7379, F1: 0.8305
Evaluation results for IoU threshold = 0.5
Eval category: test_all    , N: 34680, TP: 76170, FP: 13226, FN: 28716, Precision: 0.8521, Recall: 0.7262, F1: 0.7841
Eval category: test0_normal, N: 9621, TP: 30029, FP:  2148, FN:  2748, Precision: 0.9332, Recall: 0.9162, F1: 0.9246
Eval category: test1_crowd , N: 8113, TP: 19912, FP:  3861, FN:  8091, Precision: 0.8376, Recall: 0.7111, F1: 0.7692
Eval category: test2_hlight, N:  486, TP:  1036, FP:   289, FN:   649, Precision: 0.7819, Recall: 0.6148, F1: 0.6884
Eval category: test3_shadow, N:  930, TP:  2058, FP:   394, FN:   818, Precision: 0.8393, Recall: 0.7156, F1: 0.7725
Eval category: test4_noline, N: 4067, TP:  5879, FP:  2500, FN:  8142, Precision: 0.7016, Recall: 0.4193, F1: 0.5249
Eval category: test5_arrow , N:  890, TP:  2766, FP:   237, FN:   416, Precision: 0.9211, Recall: 0.8693, F1: 0.8944
Eval category: test6_curve , N:  422, TP:   715, FP:   116, FN:   597, Precision: 0.8604, Recall: 0.5450, F1: 0.6673
Eval category: test7_cross , N: 3122, TP:     0, FP:  1114, FN:     0, Precision: 0.0000, Recall: 0.0000, F1: 0.0000
Eval category: test8_night , N: 7029, TP: 13775, FP:  2567, FN:  7255, Precision: 0.8429, Recall: 0.6550, F1: 0.7372
Evaluation results for IoU threshold = 0.75
Eval category: test_all    , N: 34680, TP: 55988, FP: 33408, FN: 48898, Precision: 0.6263, Recall: 0.5338, F1: 0.5764
Eval category: test0_normal, N: 9621, TP: 24341, FP:  7836, FN:  8436, Precision: 0.7565, Recall: 0.7426, F1: 0.7495
Eval category: test1_crowd , N: 8113, TP: 14164, FP:  9609, FN: 13839, Precision: 0.5958, Recall: 0.5058, F1: 0.5471
Eval category: test2_hlight, N:  486, TP:   681, FP:   644, FN:  1004, Precision: 0.5140, Recall: 0.4042, F1: 0.4525
Eval category: test3_shadow, N:  930, TP:  1223, FP:  1229, FN:  1653, Precision: 0.4988, Recall: 0.4252, F1: 0.4591
Eval category: test4_noline, N: 4067, TP:  3884, FP:  4495, FN: 10137, Precision: 0.4635, Recall: 0.2770, F1: 0.3468
Eval category: test5_arrow , N:  890, TP:  2058, FP:   945, FN:  1124, Precision: 0.6853, Recall: 0.6468, F1: 0.6655
Eval category: test6_curve , N:  422, TP:   360, FP:   471, FN:   952, Precision: 0.4332, Recall: 0.2744, F1: 0.3360
Eval category: test7_cross , N: 3122, TP:     0, FP:  1114, FN:     0, Precision: 0.0000, Recall: 0.0000, F1: 0.0000
Eval category: test8_night , N: 7029, TP:  9277, FP:  7065, FN: 11753, Precision: 0.5677, Recall: 0.4411, F1: 0.4965
```
