import numpy as np
theta = [ -3.62162686e+01, -3.79772734e+01, -3.16529731e+00, -1.68213855e+02,
 1.44296782e+02,  7.36707987e+00, -4.41303114e+02,  1.42915683e+01,
-1.48313170e+00,  1.61490808e+02, -1.22517528e+02, -1.20164042e+02,
-1.06701489e+02, -3.59636492e+02, -7.04501702e+01,  1.91199470e+02,
-7.34028627e+00, -2.09438295e+02,  1.18642460e+02,  1.75405452e+02,
-2.32850568e+02, -2.32033691e+02, -2.46793705e+01,  1.79991085e+02,
 1.18883355e+02,  6.17850150e+01,  1.05080702e+02, -1.07598858e+02,
 1.26347774e+02,  3.31372635e+01, -4.74364601e+01, -1.64030958e+02,
-3.84561822e+01,  2.54441531e+01,  8.87407863e+01, -6.28739731e+00,
-9.62976070e+01, -3.31107223e+01,  4.41617533e+00,  1.12467484e+02,
 2.52731655e+01,  4.59642742e+01, -1.34361621e+00,  4.28149963e+01,
 1.03312260e+02,  4.77459541e+01,  2.63992916e+01,  1.53097709e+01,
-6.47018060e+01,  2.32525335e-01, -6.33367885e+01,  1.15025074e+01,
 1.80340011e+01,  1.16030462e+00,  7.90950900e+01, -3.11557982e+01,
-1.90875453e+01, -2.61373490e+01,  3.04086163e+01, -1.86240731e+01,
 4.97682797e+01, -4.64031648e+01,  1.31258374e+01,  1.43877232e+01,
-9.11329375e+00,  1.13240128e+02,  1.54759277e+01,  1.27784665e+01,
 1.77410386e+01, -1.15970214e+01,  1.47622740e+01, -3.33177715e+01,
 2.44730628e+01, -2.15537014e+01,  3.28332383e+00,  1.76599987e+01,
-2.54720185e+01,  6.23320848e+01, -9.43510947e+00, -8.21821391e+00,
-1.66090974e+01,  1.89815568e+00, -2.21489850e+01,  1.80319722e+01,
-4.12884308e+01,  2.18446865e+01, -3.45306924e+01, -1.22227849e+00,
 1.19202917e+01, -2.08948944e+01,  4.71021896e+01,  4.84072557e+00,
 5.17873768e+00,  1.00921396e+01,  2.31951323e+00,  1.37551320e+01,
-8.32606070e+00,  2.02744863e+01, -1.67893894e+01,  1.79534736e+01,
-9.09819031e+00, -4.21303805e+00,  1.09215316e+01, -2.65294090e+01,
 2.37605401e+01, -2.89889771e+00, -3.27571713e+00, -8.17026059e+00,
-3.81208669e+00, -1.51239031e+01,  2.03919945e+00, -2.83323605e+01,
 9.97880331e+00, -3.65238842e+01,  1.15573378e+01, -2.11051443e+01,
-4.93287643e+00,  2.21511390e-01, -1.75899576e+01, -1.53916343e+01,
 1.56794098e+00,  2.08708494e+00,  5.05563443e+00,  3.81735562e+00,
 9.65715022e+00,  1.20818728e+00,  1.50696486e+01, -5.26513500e+00,
 1.67598442e+01, -8.90649030e+00,  1.01823842e+01, -4.56385706e+00,
-6.28416643e+00,  3.44371871e+00, -2.14226790e+01, -7.27246791e+00,
-9.58332986e-01, -1.35907856e+00, -3.90095141e+00, -3.31821073e+00,
-9.30623158e+00, -2.77523679e+00, -1.77833463e+01,  1.55859924e+00,
-2.48542580e+01,  6.80981898e+00, -2.22781659e+01,  7.35384637e+00,
-1.26280770e+01, -4.61806593e+00, -9.68111053e+00, -1.17330484e+01,
-5.55559322e+01,  5.56095990e-01,  9.00795730e-01,  2.54084260e+00,
 2.70843252e+00,  6.28108931e+00,  3.38701677e+00,  9.83233129e+00,
 7.50763835e-01,  1.24425215e+01, -4.48655267e+00,  1.12560928e+01,
-5.61246200e+00,  5.68911174e+00, -2.74126277e+00, -6.07324565e+00,
-1.77711012e+00, -1.64178933e+01, -2.64770017e+01, -3.56245697e-01,
-6.14793922e-01, -1.94856975e+00, -2.16930630e+00, -5.74710191e+00,
-3.47873152e+00, -1.04553235e+01, -2.17335531e+00, -1.64643121e+01,
 2.00603949e+00, -2.12861221e+01,  5.38261847e+00, -1.42721945e+01,
 5.02209252e+00, -9.13680797e+00, -3.01967107e+00, -1.56538742e+01,
-6.96398730e+00, -7.64875151e+01]