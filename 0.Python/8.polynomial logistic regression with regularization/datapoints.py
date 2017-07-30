import numpy as np
from numpy.random import randint

def dist(v1, v2):
    dif = v1-v2
    return np.sqrt(dif.dot(dif))

def set_seed(seed):
    np.random.seed(seed)


def labeled_Points(Aamount=25, Bamount=25, 
                            Aweight=1,  Bweight=1, 
                            scale=5):
    zero_centr = np.array([-0.5,-0.5])
    A_centr = (np.random.rand(2) + zero_centr)*2*scale
    A2_centr = (np.random.rand(2) + zero_centr)*2*scale
    A3_centr = (np.random.rand(2) + zero_centr)*2*scale
    A4_centr = (np.random.rand(2) + zero_centr)*2*scale
    A5_centr = (np.random.rand(2) + zero_centr)*2*scale

    B_centr = (np.random.rand(2) + zero_centr)*2*scale
    subB2_centr = (np.random.rand(2) + zero_centr)*2*scale
    subB3_centr = (np.random.rand(2) + zero_centr)*2*scale
    subB4_centr = (np.random.rand(2) + zero_centr)*2*scale
    subB5_centr = (np.random.rand(2) + zero_centr)*2*scale
    
    get_unit_vector = lambda rad: np.c_[np.cos(rad), np.sin(rad)]

    A_magnitude    = np.random.rand(Aamount // 5)*Aweight
    A_angles       = np.random.rand(Aamount // 5)*2*np.pi
    A_unit_vectors = get_unit_vector(A_angles)
    A_deviations   = A_unit_vectors*A_magnitude[:,np.newaxis]
    A = A_deviations + A_centr


    A2_magnitude    = np.random.rand(Aamount // 5)*Aweight
    A2_angles       = np.random.rand(Aamount // 5)*2*np.pi
    A2_unit_vectors = get_unit_vector(A_angles)
    A2_deviations   = A2_unit_vectors*A2_magnitude[:,np.newaxis]
    A2 = A2_deviations + A2_centr


    A3_magnitude    = np.random.rand(Aamount // 5)*Aweight
    A3_angles       = np.random.rand(Aamount // 5)*2*np.pi
    A3_unit_vectors = get_unit_vector(A_angles)
    A3_deviations   = A3_unit_vectors*A3_magnitude[:,np.newaxis]
    A3 = A3_deviations + A3_centr

    A4_magnitude    = np.random.rand(Aamount // 5)*Aweight
    A4_angles       = np.random.rand(Aamount // 5)*2*np.pi
    A4_unit_vectors = get_unit_vector(A_angles)
    A4_deviations   = A4_unit_vectors*A4_magnitude[:,np.newaxis]
    A4 = A4_deviations + A4_centr

    A5_magnitude    = np.random.rand(Aamount // 5)*Aweight
    A5_angles       = np.random.rand(Aamount // 5)*2*np.pi
    A5_unit_vectors = get_unit_vector(A_angles)
    A5_deviations   = A5_unit_vectors*A5_magnitude[:,np.newaxis]
    A5 = A5_deviations + A5_centr

    B_magnitude    = np.random.randn(Bamount // 6)*Bweight
    B_angles       = np.random.rand(Bamount // 6)*2*np.pi
    B_unit_vectors = get_unit_vector(B_angles)
    B_deviations   = B_unit_vectors*B_magnitude[:,np.newaxis]
    B = B_deviations + B_centr

    subB1_magnitude     = np.random.randn(Bamount // 6)*(Bweight // 2)
    subB1_angles        = np.random.rand(Bamount // 6)*2*np.pi
    subB1_unit_vectors  = get_unit_vector(B_angles)
    subB1_deviations    = subB1_unit_vectors*subB1_magnitude[:,np.newaxis]
    subB1_B_center      = subB1_deviations + B[randint(0,B.shape[0]-1)]
    subB1               = subB1_B_center + B_centr

    subB2_magnitude    = np.random.randn(Bamount // 6)*Bweight
    subB2_angles       = np.random.rand(Bamount // 6)*2*np.pi
    subB2_unit_vectors = get_unit_vector(B_angles)
    subB2_deviations   = subB2_unit_vectors*subB2_magnitude[:,np.newaxis]
    subB2 = subB2_deviations + subB2_centr

    subB3_magnitude    = np.random.randn(Bamount // 6)*Bweight
    subB3_angles       = np.random.rand(Bamount // 6)*2*np.pi
    subB3_unit_vectors = get_unit_vector(B_angles)
    subB3_deviations   = subB3_unit_vectors*subB3_magnitude[:,np.newaxis]
    subB3 = subB3_deviations + subB3_centr

    subB4_magnitude    = np.random.randn(Bamount // 6)*Bweight
    subB4_angles       = np.random.rand(Bamount // 6)*2*np.pi
    subB4_unit_vectors = get_unit_vector(B_angles)
    subB4_deviations   = subB4_unit_vectors*subB4_magnitude[:,np.newaxis]
    subB4 = subB4_deviations + subB4_centr

    subB5_magnitude    = np.random.randn(Bamount // 6)*Bweight
    subB5_angles       = np.random.rand(Bamount // 6)*2*np.pi
    subB5_unit_vectors = get_unit_vector(B_angles)
    subB5_deviations   = subB5_unit_vectors*subB5_magnitude[:,np.newaxis]
    subB5 = subB5_deviations + subB5_centr
    
    B = np.row_stack((B,subB1,subB2,subB3,subB4,subB5))
    Aprep = np.row_stack((A,A2,A3,A4,A5))
    
    Aprep_lst = []
    for vector in Aprep:
        if (dist(vector, B_centr)     > Bweight*2 and
            dist(vector, subB2_centr) > Bweight*2 and
            dist(vector, subB3_centr) > Bweight*2 and
            dist(vector, subB4_centr) > Bweight*2   ):
            #
            Aprep_lst.append(vector)


    A = np.array(Aprep_lst)
    print('Aprep: ', Aprep.shape, '   A real: ', A.shape)


    A_labeled = np.c_[A, np.zeros(A.shape[0])]
    B_labeled = np.c_[B, np.ones(B.shape[0])]    


    result = np.row_stack((A_labeled,B_labeled))

    return result


if __name__ == '__main__':
    print(labeled_Points())


