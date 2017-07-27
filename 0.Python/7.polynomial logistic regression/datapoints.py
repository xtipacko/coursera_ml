import numpy as np
from numpy.random import randint
def set_seed(seed):
    np.random.seed(seed)


def labeled_Points(Aamount=25, Bamount=25, 
                            Aweight=1,  Bweight=1, 
                            scale=5):
    zero_centr = np.array([-0.5,-0.5])
    A_centr = (np.random.rand(2) + zero_centr)*2*scale
    B_centr = (np.random.rand(2) + zero_centr)*2*scale
    
    get_unit_vector = lambda rad: np.c_[np.cos(rad), np.sin(rad)]

    A_magnitude    = np.random.randn(Aamount)*Aweight
    A_angles       = np.random.rand(Aamount)*2*np.pi
    A_unit_vectors = get_unit_vector(A_angles)
    A_deviations   = A_unit_vectors*A_magnitude[:,np.newaxis]
    A = A_deviations + A_centr

    B_magnitude    = np.random.randn(Bamount // 4)*Bweight
    B_angles       = np.random.rand(Bamount // 4)*2*np.pi
    B_unit_vectors = get_unit_vector(B_angles)
    B_deviations   = B_unit_vectors*B_magnitude[:,np.newaxis]
    B = B_deviations + B_centr

    subB1_magnitude     = np.random.randn(Bamount // 4)*(Bweight // 2)
    subB1_angles        = np.random.rand(Bamount // 4)*2*np.pi
    subB1_unit_vectors  = get_unit_vector(B_angles)
    subB1_deviations    = B_unit_vectors*B_magnitude[:,np.newaxis]
    subB1_B_center      = subB1_deviations + B[randint(0,B.shape[0]-1)]
    subB1               = subB1_B_center + B_centr

    subB2_magnitude     = np.random.randn(Bamount // 4)*(Bweight // 2)
    subB2_angles        = np.random.rand(Bamount // 4)*2*np.pi
    subB2_unit_vectors  = get_unit_vector(B_angles)
    subB2_deviations    = B_unit_vectors*B_magnitude[:,np.newaxis]
    subB2_B_center      = subB2_deviations + B[randint(0,B.shape[0]-1)]
    subB2               = subB2_B_center + B_centr

    subB3_magnitude     = np.random.randn(Bamount // 4)*(Bweight // 2)
    subB3_angles        = np.random.rand(Bamount // 4)*2*np.pi
    subB3_unit_vectors  = get_unit_vector(B_angles)
    subB3_deviations    = B_unit_vectors*B_magnitude[:,np.newaxis]
    subB3_B_center      = subB3_deviations + B[randint(0,B.shape[0]-1)]
    subB3               = subB3_B_center + B_centr
    
    B = np.row_stack((B,subB1,subB2,subB3))


    A_labeled = np.c_[A, np.zeros(A.shape[0])]
    B_labeled = np.c_[B, np.ones(B.shape[0])]    


    result = np.row_stack((A_labeled,B_labeled))

    return result


if __name__ == '__main__':
    print(labeled_Points())


