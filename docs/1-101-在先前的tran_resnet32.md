⚡ ~/resnet32_ga_pso_search python train_resnet32.py \
  --run-name resnet32_baseline \
  --epochs 200 \
  --batch-size 128 \
  --lr 0.1 \
  --milestones 100,150 \
  --amp

Model: standard CIFAR-10 ResNet32
Stage channels: 16-32-64
Blocks per stage: 5
Params: 464154 (464.154K)
FLOPs : 68862592 (68.863M)

Epoch 1/200 | lr=0.100000
/teamspace/studios/this_studio/resnet32_ga_pso_search/train_resnet32.py:90: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
train:   0%|                                                                  | 0/391 [00:00<?, ?it/s]/teamspace/studios/this_studio/resnet32_ga_pso_search/train_resnet32.py:98: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=use_amp):
train_loss=1.7402 train_acc=35.22 | test_loss=1.4438 test_acc=47.68 | time=6.8s                       
Saved new best checkpoint: acc=47.68%

Epoch 2/200 | lr=0.100000
train_loss=1.2212 train_acc=55.81 | test_loss=1.3092 test_acc=55.07 | time=4.8s                       
Saved new best checkpoint: acc=55.07%

Epoch 3/200 | lr=0.100000
train_loss=0.9695 train_acc=65.74 | test_loss=1.0911 test_acc=62.17 | time=4.7s                       
Saved new best checkpoint: acc=62.17%

Epoch 4/200 | lr=0.100000
train_loss=0.8110 train_acc=71.46 | test_loss=0.8605 test_acc=70.57 | time=4.8s                       
Saved new best checkpoint: acc=70.57%

Epoch 5/200 | lr=0.100000
train_loss=0.6946 train_acc=75.81 | test_loss=0.8324 test_acc=73.16 | time=4.8s                       
Saved new best checkpoint: acc=73.16%

Epoch 6/200 | lr=0.100000
train_loss=0.6227 train_acc=78.22 | test_loss=0.6832 test_acc=77.06 | time=4.8s                       
Saved new best checkpoint: acc=77.06%

Epoch 7/200 | lr=0.100000
train_loss=0.5695 train_acc=80.18 | test_loss=0.6282 test_acc=78.98 | time=4.9s                       
Saved new best checkpoint: acc=78.98%

Epoch 8/200 | lr=0.100000
train_loss=0.5325 train_acc=81.64 | test_loss=0.6148 test_acc=79.40 | time=4.8s                       
Saved new best checkpoint: acc=79.40%

Epoch 9/200 | lr=0.100000
train_loss=0.4990 train_acc=82.74 | test_loss=0.5642 test_acc=80.52 | time=5.6s                       
Saved new best checkpoint: acc=80.52%

Epoch 10/200 | lr=0.100000
train_loss=0.4655 train_acc=83.99 | test_loss=0.5546 test_acc=81.16 | time=5.8s                       
Saved new best checkpoint: acc=81.16%

Epoch 11/200 | lr=0.100000
train_loss=0.4523 train_acc=84.36 | test_loss=0.5419 test_acc=81.92 | time=6.3s
Saved new best checkpoint: acc=81.92%

Epoch 12/200 | lr=0.100000
train_loss=0.4244 train_acc=85.23 | test_loss=0.4995 test_acc=82.86 | time=6.1s
Saved new best checkpoint: acc=82.86%

Epoch 13/200 | lr=0.100000
train_loss=0.4100 train_acc=85.67 | test_loss=0.4872 test_acc=83.72 | time=6.2s
Saved new best checkpoint: acc=83.72%

Epoch 14/200 | lr=0.100000
train_loss=0.3945 train_acc=86.47 | test_loss=0.7322 test_acc=77.37 | time=6.0s

Epoch 15/200 | lr=0.100000
train_loss=0.3826 train_acc=86.81 | test_loss=0.5877 test_acc=80.86 | time=6.1s                       

Epoch 16/200 | lr=0.100000
train_loss=0.3719 train_acc=87.04 | test_loss=0.5560 test_acc=82.10 | time=6.1s                       

Epoch 17/200 | lr=0.100000
train_loss=0.3636 train_acc=87.31 | test_loss=0.6185 test_acc=79.97 | time=6.1s                       

Epoch 18/200 | lr=0.100000
train_loss=0.3521 train_acc=87.68 | test_loss=0.4309 test_acc=85.64 | time=6.2s                       
Saved new best checkpoint: acc=85.64%

Epoch 19/200 | lr=0.100000
train_loss=0.3445 train_acc=87.95 | test_loss=0.4667 test_acc=84.77 | time=6.1s                       

Epoch 20/200 | lr=0.100000
train_loss=0.3313 train_acc=88.53 | test_loss=0.4910 test_acc=84.07 | time=6.1s                       

Epoch 21/200 | lr=0.100000
train_loss=0.3306 train_acc=88.41 | test_loss=0.5283 test_acc=83.92 | time=6.1s                       

Epoch 22/200 | lr=0.100000
train_loss=0.3210 train_acc=88.84 | test_loss=0.5370 test_acc=83.57 | time=6.3s                       

Epoch 23/200 | lr=0.100000
train_loss=0.3132 train_acc=89.11 | test_loss=0.4853 test_acc=84.93 | time=6.1s                       

Epoch 24/200 | lr=0.100000
train_loss=0.3108 train_acc=89.03 | test_loss=0.4641 test_acc=84.62 | time=6.1s                       

Epoch 25/200 | lr=0.100000
train_loss=0.2969 train_acc=89.74 | test_loss=0.5500 test_acc=83.16 | time=6.3s                       

Epoch 26/200 | lr=0.100000
train_loss=0.2968 train_acc=89.66 | test_loss=0.4700 test_acc=85.07 | time=6.2s                       

Epoch 27/200 | lr=0.100000
train_loss=0.2901 train_acc=89.88 | test_loss=0.4767 test_acc=84.58 | time=6.0s                       

Epoch 28/200 | lr=0.100000
train_loss=0.2865 train_acc=90.01 | test_loss=0.4835 test_acc=84.83 | time=6.1s                       

Epoch 29/200 | lr=0.100000
train_loss=0.2798 train_acc=90.37 | test_loss=0.4954 test_acc=84.71 | time=5.9s                       

Epoch 30/200 | lr=0.100000
train_loss=0.2799 train_acc=90.23 | test_loss=0.4772 test_acc=84.63 | time=6.1s

Epoch 31/200 | lr=0.100000
train_loss=0.2784 train_acc=90.33 | test_loss=0.4496 test_acc=85.53 | time=6.0s

Epoch 32/200 | lr=0.100000
train_loss=0.2682 train_acc=90.74 | test_loss=0.4497 test_acc=85.57 | time=6.2s

Epoch 33/200 | lr=0.100000
train_loss=0.2658 train_acc=90.77 | test_loss=0.4417 test_acc=86.10 | time=6.1s
Saved new best checkpoint: acc=86.10%

Epoch 34/200 | lr=0.100000
train_loss=0.2642 train_acc=90.81 | test_loss=0.4192 test_acc=86.64 | time=6.2s
Saved new best checkpoint: acc=86.64%

Epoch 35/200 | lr=0.100000
train_loss=0.2619 train_acc=90.86 | test_loss=0.4575 test_acc=85.28 | time=6.1s                       

Epoch 36/200 | lr=0.100000
train_loss=0.2563 train_acc=91.12 | test_loss=0.4551 test_acc=85.72 | time=6.2s                       

Epoch 37/200 | lr=0.100000
train_loss=0.2556 train_acc=91.08 | test_loss=0.4242 test_acc=86.35 | time=6.1s                       

Epoch 38/200 | lr=0.100000
train_loss=0.2549 train_acc=91.02 | test_loss=0.4305 test_acc=85.97 | time=6.1s                       

Epoch 39/200 | lr=0.100000
train_loss=0.2496 train_acc=91.39 | test_loss=0.4090 test_acc=86.89 | time=6.2s                       
Saved new best checkpoint: acc=86.89%

Epoch 40/200 | lr=0.100000
train_loss=0.2492 train_acc=91.31 | test_loss=0.4186 test_acc=86.83 | time=6.2s                       

Epoch 41/200 | lr=0.100000
train_loss=0.2444 train_acc=91.49 | test_loss=0.3825 test_acc=87.87 | time=6.1s                       
Saved new best checkpoint: acc=87.87%

Epoch 42/200 | lr=0.100000
train_loss=0.2446 train_acc=91.46 | test_loss=0.3998 test_acc=87.19 | time=6.3s                       

Epoch 43/200 | lr=0.100000
train_loss=0.2390 train_acc=91.71 | test_loss=0.4084 test_acc=86.53 | time=6.0s                       

Epoch 44/200 | lr=0.100000
train_loss=0.2417 train_acc=91.63 | test_loss=0.4227 test_acc=86.77 | time=6.2s                       

Epoch 45/200 | lr=0.100000
train_loss=0.2312 train_acc=91.82 | test_loss=0.3843 test_acc=87.62 | time=6.1s                       

Epoch 46/200 | lr=0.100000
train_loss=0.2319 train_acc=91.88 | test_loss=0.4420 test_acc=86.02 | time=6.1s                       

Epoch 47/200 | lr=0.100000
train_loss=0.2295 train_acc=91.96 | test_loss=0.5070 test_acc=84.93 | time=6.1s                       

Epoch 48/200 | lr=0.100000
train_loss=0.2326 train_acc=91.88 | test_loss=0.4913 test_acc=84.74 | time=6.1s                       

Epoch 49/200 | lr=0.100000
train_loss=0.2249 train_acc=92.10 | test_loss=0.4347 test_acc=86.44 | time=6.2s                       

Epoch 50/200 | lr=0.100000
train_loss=0.2221 train_acc=92.23 | test_loss=0.4846 test_acc=84.90 | time=6.4s                       

Epoch 51/200 | lr=0.100000
train_loss=0.2266 train_acc=92.09 | test_loss=0.4441 test_acc=85.71 | time=6.1s                       

Epoch 52/200 | lr=0.100000
train_loss=0.2199 train_acc=92.24 | test_loss=0.4820 test_acc=85.45 | time=6.1s                       

Epoch 53/200 | lr=0.100000
train_loss=0.2252 train_acc=92.14 | test_loss=0.4428 test_acc=86.47 | time=6.1s                       

Epoch 54/200 | lr=0.100000
train_loss=0.2249 train_acc=92.14 | test_loss=0.4212 test_acc=87.18 | time=6.2s                       

Epoch 55/200 | lr=0.100000
train_loss=0.2174 train_acc=92.40 | test_loss=0.3908 test_acc=87.31 | time=6.0s                       

Epoch 57/200 | lr=0.100000
train_loss=0.2123 train_acc=92.56 | test_loss=0.3685 test_acc=88.24 | time=6.0s
Saved new best checkpoint: acc=88.24%

Epoch 58/200 | lr=0.100000
train_loss=0.2201 train_acc=92.26 | test_loss=0.4250 test_acc=86.94 | time=6.2s

Epoch 59/200 | lr=0.100000
train_loss=0.2147 train_acc=92.23 | test_loss=0.4632 test_acc=86.27 | time=6.3s

Epoch 60/200 | lr=0.100000
train_loss=0.2173 train_acc=92.35 | test_loss=0.4446 test_acc=86.40 | time=6.2s

Epoch 61/200 | lr=0.100000
train_loss=0.2116 train_acc=92.62 | test_loss=0.4466 test_acc=86.46 | time=6.2s

Epoch 62/200 | lr=0.100000
train_loss=0.2109 train_acc=92.53 | test_loss=0.5626 test_acc=83.97 | time=6.2s

Epoch 63/200 | lr=0.100000
train_loss=0.2108 train_acc=92.54 | test_loss=0.4740 test_acc=85.56 | time=6.1s

Epoch 64/200 | lr=0.100000
train_loss=0.2030 train_acc=92.90 | test_loss=0.3821 test_acc=88.36 | time=6.2s
Saved new best checkpoint: acc=88.36%

Epoch 65/200 | lr=0.100000
train_loss=0.2030 train_acc=92.90 | test_loss=0.4040 test_acc=87.82 | time=6.2s                       

Epoch 66/200 | lr=0.100000
train_loss=0.2043 train_acc=92.88 | test_loss=0.4007 test_acc=87.31 | time=7.0s                       

Epoch 67/200 | lr=0.100000
train_loss=0.2056 train_acc=92.85 | test_loss=0.4058 test_acc=87.13 | time=6.3s                       

Epoch 68/200 | lr=0.100000
train_loss=0.1985 train_acc=93.07 | test_loss=0.3950 test_acc=87.77 | time=10.4s                      

Epoch 69/200 | lr=0.100000
train_loss=0.2030 train_acc=92.86 | test_loss=0.4396 test_acc=86.70 | time=6.1s                       

Epoch 70/200 | lr=0.100000
train_loss=0.2046 train_acc=92.88 | test_loss=0.3966 test_acc=88.11 | time=6.2s                       

Epoch 71/200 | lr=0.100000
train_loss=0.2026 train_acc=92.95 | test_loss=0.4091 test_acc=87.47 | time=6.2s                       

Epoch 72/200 | lr=0.100000
train_loss=0.2012 train_acc=92.94 | test_loss=0.4274 test_acc=87.74 | time=6.1s                       

Epoch 73/200 | lr=0.100000
train_loss=0.2026 train_acc=92.95 | test_loss=0.4500 test_acc=86.55 | time=6.1s                       

Epoch 74/200 | lr=0.100000
train_loss=0.1975 train_acc=93.13 | test_loss=0.4573 test_acc=86.21 | time=6.2s                       

Epoch 75/200 | lr=0.100000
train_loss=0.1990 train_acc=93.05 | test_loss=0.4495 test_acc=86.96 | time=6.1s                       

Epoch 76/200 | lr=0.100000
train_loss=0.1949 train_acc=93.24 | test_loss=0.5547 test_acc=84.46 | time=6.1s                       

Epoch 77/200 | lr=0.100000
train_loss=0.1895 train_acc=93.35 | test_loss=0.3773 test_acc=88.48 | time=6.4s                       
Saved new best checkpoint: acc=88.48%

Epoch 78/200 | lr=0.100000
train_loss=0.1962 train_acc=93.05 | test_loss=0.3921 test_acc=88.08 | time=6.1s                       

Epoch 79/200 | lr=0.100000
train_loss=0.1928 train_acc=93.28 | test_loss=0.3983 test_acc=87.66 | time=6.2s                       

Epoch 80/200 | lr=0.100000
train_loss=0.1961 train_acc=93.13 | test_loss=0.4092 test_acc=87.73 | time=6.1s                       

Epoch 81/200 | lr=0.100000
train_loss=0.1923 train_acc=93.34 | test_loss=0.3741 test_acc=88.36 | time=6.1s                       

Epoch 82/200 | lr=0.100000
train_loss=0.1885 train_acc=93.43 | test_loss=0.4786 test_acc=86.10 | time=6.2s                       

Epoch 83/200 | lr=0.100000
train_loss=0.1937 train_acc=93.29 | test_loss=0.4227 test_acc=87.04 | time=6.2s                       

Epoch 84/200 | lr=0.100000
train_loss=0.2002 train_acc=92.90 | test_loss=0.3638 test_acc=88.24 | time=6.2s                       

Epoch 85/200 | lr=0.100000
train_loss=0.1903 train_acc=93.34 | test_loss=0.6559 test_acc=82.31 | time=6.5s                       

Epoch 86/200 | lr=0.100000
train_loss=0.1902 train_acc=93.22 | test_loss=0.3871 test_acc=87.85 | time=6.1s                       

Epoch 87/200 | lr=0.100000
train_loss=0.1913 train_acc=93.15 | test_loss=0.4516 test_acc=86.99 | time=6.2s                       

Epoch 88/200 | lr=0.100000
train_loss=0.1945 train_acc=93.13 | test_loss=0.3741 test_acc=88.67 | time=6.3s                       
Saved new best checkpoint: acc=88.67%

Epoch 89/200 | lr=0.100000
train_loss=0.1885 train_acc=93.38 | test_loss=0.4615 test_acc=86.29 | time=5.6s                       

Epoch 90/200 | lr=0.100000
train_loss=0.1837 train_acc=93.53 | test_loss=0.4699 test_acc=85.93 | time=4.8s                       

Epoch 91/200 | lr=0.100000
train_loss=0.1883 train_acc=93.45 | test_loss=0.3876 test_acc=88.40 | time=4.8s                       

Epoch 92/200 | lr=0.100000
train_loss=0.1802 train_acc=93.71 | test_loss=0.4406 test_acc=86.67 | time=4.9s                       

Epoch 93/200 | lr=0.100000
train_loss=0.1877 train_acc=93.53 | test_loss=0.3756 test_acc=88.60 | time=4.7s                       

Epoch 94/200 | lr=0.100000
train_loss=0.1842 train_acc=93.61 | test_loss=0.4231 test_acc=86.64 | time=4.8s                       

Epoch 95/200 | lr=0.100000
train_loss=0.1907 train_acc=93.37 | test_loss=0.4682 test_acc=86.05 | time=4.8s                       

Epoch 96/200 | lr=0.100000
train_loss=0.1847 train_acc=93.49 | test_loss=0.4624 test_acc=86.05 | time=4.7s                       

Epoch 97/200 | lr=0.100000
train_loss=0.1840 train_acc=93.48 | test_loss=0.4296 test_acc=87.02 | time=4.8s                       

Epoch 98/200 | lr=0.100000
train_loss=0.1874 train_acc=93.48 | test_loss=0.5062 test_acc=85.20 | time=4.9s                       

Epoch 99/200 | lr=0.100000
train_loss=0.1856 train_acc=93.66 | test_loss=0.4939 test_acc=85.25 | time=4.9s                       

Epoch 100/200 | lr=0.100000
train_loss=0.1832 train_acc=93.46 | test_loss=0.3919 test_acc=88.38 | time=4.8s                       

Epoch 101/200 | lr=0.010000
train_loss=0.1032 train_acc=96.52 | test_loss=0.2587 test_acc=91.93 | time=4.8s                       
Saved new best checkpoint: acc=91.93%

Epoch 102/200 | lr=0.010000
train_loss=0.0761 train_acc=97.44 | test_loss=0.2574 test_acc=91.97 | time=4.8s                       
Saved new best checkpoint: acc=91.97%

Epoch 103/200 | lr=0.010000
train_loss=0.0656 train_acc=97.86 | test_loss=0.2624 test_acc=92.26 | time=4.7s                       
Saved new best checkpoint: acc=92.26%

Epoch 104/200 | lr=0.010000
train_loss=0.0599 train_acc=98.01 | test_loss=0.2602 test_acc=92.32 | time=4.7s                       
Saved new best checkpoint: acc=92.32%

Epoch 105/200 | lr=0.010000
train_loss=0.0526 train_acc=98.35 | test_loss=0.2632 test_acc=92.24 | time=4.7s                       

Epoch 106/200 | lr=0.010000
train_loss=0.0499 train_acc=98.40 | test_loss=0.2638 test_acc=92.18 | time=4.8s                       

Epoch 107/200 | lr=0.010000
train_loss=0.0464 train_acc=98.52 | test_loss=0.2664 test_acc=92.33 | time=4.8s                       
Saved new best checkpoint: acc=92.33%

Epoch 108/200 | lr=0.010000
train_loss=0.0454 train_acc=98.58 | test_loss=0.2689 test_acc=92.38 | time=4.7s                       
Saved new best checkpoint: acc=92.38%

Epoch 109/200 | lr=0.010000
train_loss=0.0404 train_acc=98.69 | test_loss=0.2676 test_acc=92.33 | time=4.8s                       

Epoch 110/200 | lr=0.010000
train_loss=0.0386 train_acc=98.75 | test_loss=0.2779 test_acc=92.13 | time=4.9s                       

Epoch 111/200 | lr=0.010000
train_loss=0.0367 train_acc=98.83 | test_loss=0.2752 test_acc=92.35 | time=4.8s                       

Epoch 112/200 | lr=0.010000
train_loss=0.0345 train_acc=98.91 | test_loss=0.2767 test_acc=92.43 | time=4.8s                       
Saved new best checkpoint: acc=92.43%

Epoch 113/200 | lr=0.010000
train_loss=0.0334 train_acc=98.99 | test_loss=0.2763 test_acc=92.47 | time=4.7s                       
Saved new best checkpoint: acc=92.47%

Epoch 114/200 | lr=0.010000
train_loss=0.0308 train_acc=99.06 | test_loss=0.2825 test_acc=92.50 | time=4.8s                       
Saved new best checkpoint: acc=92.50%

Epoch 115/200 | lr=0.010000
train_loss=0.0308 train_acc=99.02 | test_loss=0.2871 test_acc=92.41 | time=4.8s                       

Epoch 116/200 | lr=0.010000
train_loss=0.0278 train_acc=99.20 | test_loss=0.2869 test_acc=92.48 | time=4.8s                       

Epoch 117/200 | lr=0.010000
train_loss=0.0269 train_acc=99.15 | test_loss=0.2899 test_acc=92.39 | time=4.9s                       

Epoch 118/200 | lr=0.010000
train_loss=0.0261 train_acc=99.20 | test_loss=0.2955 test_acc=92.52 | time=4.8s                       
Saved new best checkpoint: acc=92.52%

Epoch 119/200 | lr=0.010000
train_loss=0.0254 train_acc=99.22 | test_loss=0.2964 test_acc=92.42 | time=4.8s                       

Epoch 120/200 | lr=0.010000
train_loss=0.0247 train_acc=99.24 | test_loss=0.2944 test_acc=92.59 | time=4.8s                       
Saved new best checkpoint: acc=92.59%

Epoch 121/200 | lr=0.010000
train_loss=0.0249 train_acc=99.20 | test_loss=0.2967 test_acc=92.70 | time=4.9s                       
Saved new best checkpoint: acc=92.70%

Epoch 122/200 | lr=0.010000
train_loss=0.0230 train_acc=99.30 | test_loss=0.2991 test_acc=92.37 | time=4.7s                       

Epoch 123/200 | lr=0.010000
train_loss=0.0221 train_acc=99.36 | test_loss=0.3046 test_acc=92.31 | time=4.7s                       

Epoch 124/200 | lr=0.010000
train_loss=0.0218 train_acc=99.34 | test_loss=0.3043 test_acc=92.39 | time=4.9s                       

Epoch 125/200 | lr=0.010000
train_loss=0.0198 train_acc=99.40 | test_loss=0.3064 test_acc=92.46 | time=5.3s

Epoch 126/200 | lr=0.010000
train_loss=0.0198 train_acc=99.45 | test_loss=0.3031 test_acc=92.56 | time=4.7s

Epoch 127/200 | lr=0.010000
train_loss=0.0192 train_acc=99.44 | test_loss=0.3146 test_acc=92.36 | time=4.7s

Epoch 128/200 | lr=0.010000
train_loss=0.0181 train_acc=99.46 | test_loss=0.3119 test_acc=92.23 | time=4.9s

Epoch 129/200 | lr=0.010000
train_loss=0.0178 train_acc=99.47 | test_loss=0.3109 test_acc=92.45 | time=4.8s

Epoch 130/200 | lr=0.010000
train_loss=0.0181 train_acc=99.49 | test_loss=0.3142 test_acc=92.29 | time=4.8s

Epoch 131/200 | lr=0.010000
train_loss=0.0168 train_acc=99.56 | test_loss=0.3173 test_acc=92.39 | time=4.8s                       

Epoch 132/200 | lr=0.010000
train_loss=0.0168 train_acc=99.54 | test_loss=0.3156 test_acc=92.47 | time=4.8s                       

Epoch 133/200 | lr=0.010000
train_loss=0.0159 train_acc=99.58 | test_loss=0.3124 test_acc=92.63 | time=4.8s                       

Epoch 134/200 | lr=0.010000
train_loss=0.0164 train_acc=99.55 | test_loss=0.3223 test_acc=92.54 | time=4.7s                       

Epoch 135/200 | lr=0.010000
train_loss=0.0156 train_acc=99.57 | test_loss=0.3216 test_acc=92.37 | time=4.7s                       

Epoch 136/200 | lr=0.010000
train_loss=0.0152 train_acc=99.57 | test_loss=0.3207 test_acc=92.58 | time=4.8s                       

Epoch 137/200 | lr=0.010000
train_loss=0.0153 train_acc=99.59 | test_loss=0.3214 test_acc=92.57 | time=4.8s                       

Epoch 138/200 | lr=0.010000
train_loss=0.0149 train_acc=99.59 | test_loss=0.3192 test_acc=92.66 | time=4.7s                       

Epoch 139/200 | lr=0.010000
train_loss=0.0152 train_acc=99.57 | test_loss=0.3100 test_acc=92.62 | time=5.2s                       

Epoch 140/200 | lr=0.010000
train_loss=0.0142 train_acc=99.61 | test_loss=0.3192 test_acc=92.48 | time=4.9s                       

Epoch 141/200 | lr=0.010000
train_loss=0.0139 train_acc=99.61 | test_loss=0.3201 test_acc=92.56 | time=4.9s                       

Epoch 142/200 | lr=0.010000
train_loss=0.0141 train_acc=99.61 | test_loss=0.3301 test_acc=92.52 | time=4.7s                       

Epoch 143/200 | lr=0.010000
train_loss=0.0142 train_acc=99.59 | test_loss=0.3296 test_acc=92.55 | time=4.9s                       

Epoch 144/200 | lr=0.010000
train_loss=0.0132 train_acc=99.64 | test_loss=0.3239 test_acc=92.60 | time=5.0s                       

Epoch 145/200 | lr=0.010000
train_loss=0.0136 train_acc=99.60 | test_loss=0.3426 test_acc=92.48 | time=5.0s

Epoch 146/200 | lr=0.010000
train_loss=0.0138 train_acc=99.60 | test_loss=0.3397 test_acc=92.50 | time=5.0s

Epoch 147/200 | lr=0.010000
train_loss=0.0119 train_acc=99.68 | test_loss=0.3368 test_acc=92.35 | time=4.8s

Epoch 148/200 | lr=0.010000
train_loss=0.0126 train_acc=99.65 | test_loss=0.3288 test_acc=92.64 | time=4.8s

Epoch 149/200 | lr=0.010000
train_loss=0.0117 train_acc=99.67 | test_loss=0.3338 test_acc=92.38 | time=4.9s

Epoch 150/200 | lr=0.010000
train_loss=0.0121 train_acc=99.68 | test_loss=0.3364 test_acc=92.61 | time=4.8s                       

Epoch 151/200 | lr=0.001000
train_loss=0.0106 train_acc=99.72 | test_loss=0.3289 test_acc=92.58 | time=4.9s                       

Epoch 152/200 | lr=0.001000
train_loss=0.0104 train_acc=99.74 | test_loss=0.3294 test_acc=92.71 | time=4.8s                       
Saved new best checkpoint: acc=92.71%

Epoch 153/200 | lr=0.001000
train_loss=0.0093 train_acc=99.77 | test_loss=0.3251 test_acc=92.73 | time=4.8s                       
Saved new best checkpoint: acc=92.73%

Epoch 154/200 | lr=0.001000
train_loss=0.0092 train_acc=99.77 | test_loss=0.3267 test_acc=92.66 | time=4.8s                       

Epoch 155/200 | lr=0.001000
train_loss=0.0086 train_acc=99.79 | test_loss=0.3265 test_acc=92.62 | time=4.8s                       

Epoch 156/200 | lr=0.001000
train_loss=0.0077 train_acc=99.83 | test_loss=0.3248 test_acc=92.62 | time=4.8s                       

Epoch 157/200 | lr=0.001000
train_loss=0.0090 train_acc=99.77 | test_loss=0.3251 test_acc=92.68 | time=4.9s                       

Epoch 158/200 | lr=0.001000
train_loss=0.0086 train_acc=99.80 | test_loss=0.3241 test_acc=92.54 | time=4.8s                       

Epoch 159/200 | lr=0.001000
train_loss=0.0080 train_acc=99.82 | test_loss=0.3261 test_acc=92.79 | time=4.8s                       
Saved new best checkpoint: acc=92.79%

Epoch 160/200 | lr=0.001000
train_loss=0.0075 train_acc=99.84 | test_loss=0.3269 test_acc=92.64 | time=4.7s                       

Epoch 161/200 | lr=0.001000
train_loss=0.0077 train_acc=99.82 | test_loss=0.3206 test_acc=92.77 | time=4.8s                       

Epoch 162/200 | lr=0.001000
train_loss=0.0075 train_acc=99.85 | test_loss=0.3228 test_acc=92.77 | time=4.7s                       

Epoch 163/200 | lr=0.001000
train_loss=0.0077 train_acc=99.83 | test_loss=0.3238 test_acc=92.67 | time=4.9s                       

Epoch 164/200 | lr=0.001000
train_loss=0.0080 train_acc=99.82 | test_loss=0.3231 test_acc=92.76 | time=4.9s                       

Epoch 165/200 | lr=0.001000
train_loss=0.0082 train_acc=99.81 | test_loss=0.3232 test_acc=92.69 | time=4.8s                       

Epoch 166/200 | lr=0.001000
train_loss=0.0082 train_acc=99.78 | test_loss=0.3260 test_acc=92.65 | time=4.7s                       

Epoch 167/200 | lr=0.001000
train_loss=0.0077 train_acc=99.83 | test_loss=0.3242 test_acc=92.61 | time=4.7s                       

Epoch 168/200 | lr=0.001000
train_loss=0.0076 train_acc=99.83 | test_loss=0.3268 test_acc=92.66 | time=4.7s                       

Epoch 169/200 | lr=0.001000
train_loss=0.0078 train_acc=99.83 | test_loss=0.3254 test_acc=92.70 | time=4.9s                       

Epoch 170/200 | lr=0.001000
train_loss=0.0073 train_acc=99.84 | test_loss=0.3226 test_acc=92.71 | time=4.8s                       

Epoch 171/200 | lr=0.001000
train_loss=0.0077 train_acc=99.84 | test_loss=0.3230 test_acc=92.71 | time=4.8s                       

Epoch 172/200 | lr=0.001000
train_loss=0.0071 train_acc=99.85 | test_loss=0.3236 test_acc=92.64 | time=4.8s                       

Epoch 173/200 | lr=0.001000
train_loss=0.0073 train_acc=99.83 | test_loss=0.3266 test_acc=92.74 | time=4.8s                       

Epoch 174/200 | lr=0.001000
train_loss=0.0075 train_acc=99.85 | test_loss=0.3227 test_acc=92.76 | time=4.7s                       

Epoch 175/200 | lr=0.001000
train_loss=0.0074 train_acc=99.86 | test_loss=0.3231 test_acc=92.74 | time=4.8s                       

Epoch 176/200 | lr=0.001000
train_loss=0.0072 train_acc=99.86 | test_loss=0.3276 test_acc=92.57 | time=4.8s                       

Epoch 177/200 | lr=0.001000
train_loss=0.0068 train_acc=99.87 | test_loss=0.3241 test_acc=92.80 | time=4.7s                       
Saved new best checkpoint: acc=92.80%

Epoch 178/200 | lr=0.001000
train_loss=0.0068 train_acc=99.87 | test_loss=0.3247 test_acc=92.79 | time=4.7s                       

Epoch 179/200 | lr=0.00010
train_loss=0.0071 train_acc=99.85 | test_loss=0.3259 test_acc=92.81 | time=4.8s                       
Saved new best checkpoint: acc=92.81%

Epoch 180/200 | lr=0.001000
train_loss=0.0067 train_acc=99.88 | test_loss=0.3267 test_acc=92.74 | time=4.8s                       

Epoch 181/200 | lr=0.001000
train_loss=0.0071 train_acc=99.86 | test_loss=0.3285 test_acc=92.73 | time=4.8s                       

Epoch 182/200 | lr=0.001000
train_loss=0.0068 train_acc=99.87 | test_loss=0.3264 test_acc=92.84 | time=4.8s                       
Saved new best checkpoint: acc=92.84%

Epoch 183/200 | lr=0.001000
train_loss=0.0069 train_acc=99.85 | test_loss=0.3250 test_acc=92.68 | time=4.8s                       

Epoch 184/200 | lr=0.001000
train_loss=0.0070 train_acc=99.87 | test_loss=0.3254 test_acc=92.73 | time=4.8s                       

Epoch 185/200 | lr=0.001000
train_loss=0.0072 train_acc=99.85 | test_loss=0.3248 test_acc=92.82 | time=4.8s                       

Epoch 186/200 | lr=0.001000
train_loss=0.0070 train_acc=99.86 | test_loss=0.3260 test_acc=92.78 | time=4.8s                       

Epoch 187/200 | lr=0.001000
train_loss=0.0063 train_acc=99.87 | test_loss=0.3267 test_acc=92.70 | time=4.8s                       

Epoch 188/200 | lr=0.001000
train_loss=0.0072 train_acc=99.84 | test_loss=0.3310 test_acc=92.69 | time=4.8s                       

Epoch 189/200 | lr=0.001000
train_loss=0.0067 train_acc=99.89 | test_loss=0.3266 test_acc=92.69 | time=4.7s                       

Epoch 190/200 | lr=0.001000
train_loss=0.0069 train_acc=99.85 | test_loss=0.3287 test_acc=92.53 | time=4.8s                       

Epoch 191/200 | lr=0.001000
train_loss=0.0070 train_acc=99.85 | test_loss=0.3273 test_acc=92.74 | time=4.8s                       

Epoch 192/200 | lr=0.001000
train_loss=0.0066 train_acc=99.87 | test_loss=0.3266 test_acc=92.74 | time=4.7s                       

Epoch 193/200 | lr=0.001000
train_loss=0.0071 train_acc=99.83 | test_loss=0.3259 test_acc=92.72 | time=4.8s                       

Epoch 194/200 | lr=0.001000
train_loss=0.0068 train_acc=99.87 | test_loss=0.3228 test_acc=92.84 | time=4.8s                       

Epoch 195/200 | lr=0.001000
train_loss=0.0063 train_acc=99.88 | test_loss=0.3286 test_acc=92.73 | time=4.8s                       

Epoch 196/200 | lr=0.001000
train_loss=0.0064 train_acc=99.90 | test_loss=0.3253 test_acc=92.83 | time=4.7s                       

Epoch 197/200 | lr=0.001000
train_loss=0.0066 train_acc=99.86 | test_loss=0.3289 test_acc=92.72 | time=4.7s                       

Epoch 198/200 | lr=0.001000
train_loss=0.0067 train_acc=99.88 | test_loss=0.3250 test_acc=92.79 | time=4.8s                       

Epoch 199/200 | lr=0.001000
train_loss=0.0065 train_acc=99.86 | test_loss=0.3249 test_acc=92.81 | time=4.8s                       

Epoch 200/200 | lr=0.001000
train_loss=0.0062 train_acc=99.90 | test_loss=0.3250 test_acc=92.83 | time=4.7s                       

Done.
{
  "run_name": "resnet32_baseline",
  "model": "ResNet32",
  "dataset": "CIFAR-10",
  "stage_channels": [16,32,64],
  "blocks_per_stage": 5,
  "epochs": 200,
  "best_acc": 92.84,
  "best_epoch": 182,
  "params": 464154,
  "flops": 68862592,
  "total_train_time_sec": 1080.4495613574982,
  "checkpoint_best": "runs/resnet32_baseline/best.pt",
  "checkpoint_last": "runs/resnet32_baseline/last.pt"
}