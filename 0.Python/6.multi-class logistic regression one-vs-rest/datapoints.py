import numpy as np

def set_seed(seed):
    np.random.seed(seed)



def labeled_three_classPoints(Aamount=25, Bamount=25, Camount=25,
                            Aweight=1,  Bweight=1,  Cweight=1,
                            scale=5):
    zero_centr = np.array([-0.5,-0.5])
    A_centr = (np.random.rand(2) + zero_centr)*2*scale
    B_centr = (np.random.rand(2) + zero_centr)*2*scale
    C_centr = (np.random.rand(2) + zero_centr)*2*scale
    
    get_unit_vector = lambda rad: np.c_[np.cos(rad), np.sin(rad)]

    A_magnitude    = np.random.randn(Aamount)*Aweight
    A_angles       = np.random.rand(Aamount)*2*np.pi
    A_unit_vectors = get_unit_vector(A_angles)
    A_deviations   = A_unit_vectors*A_magnitude[:,np.newaxis]
    A = A_deviations + A_centr

    B_magnitude    = np.random.randn(Bamount)*Bweight
    B_angles       = np.random.rand(Bamount)*2*np.pi
    B_unit_vectors = get_unit_vector(B_angles)
    B_deviations   = B_unit_vectors*B_magnitude[:,np.newaxis]
    B = B_deviations + B_centr

    C_magnitude    = np.random.randn(Camount)*Cweight
    C_angles       = np.random.rand(Camount)*2*np.pi
    C_unit_vectors = get_unit_vector(C_angles)
    C_deviations   = C_unit_vectors*C_magnitude[:,np.newaxis]
    C = C_deviations + C_centr

    A_labeled = np.c_[A, np.zeros(A.shape[0])]
    B_labeled = np.c_[B, np.ones(B.shape[0])]    
    C_labeled = np.c_[C, np.ones(C.shape[0])*2]

    result = np.row_stack((A_labeled,B_labeled,C_labeled))

    return result


if __name__ == '__main__':
    print(labeled_two_classPoints())


