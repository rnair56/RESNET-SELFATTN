import sys
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as k


def LossCse673(inmat, y_labels, batch_size,  alpha, beta, lambda_1, no_classes=98):
    ##Assert shape
    # if inmat.shape != (batch_size, 512):
    #     sys.exit("expected batch_size, 512 shape, recieved", (batch_size, 512))
    ##the sum of squares for each vector in the matrix (similarity matrix)
    ##BxB
    t_squared_vectors = tf.matmul(inmat, inmat, transpose_b=True)
    ##The diagonal of the sum of the squares (gives B x v1 dot v1.T) ||a||
    diagonals = tf.linalg.tensor_diag_part(t_squared_vectors)
    ##L2 Norm
    l2_norm = tf.math.sqrt(diagonals)
    # print("shape l2 norm", l2_norm.shape)
    ##
    l2_norm = tf.expand_dims(l2_norm, axis=1)
    # print("shape l2 norm ",l2_norm.shape)

    inmatNormalised = inmat / l2_norm
    # print("inmat")
    # print(inmat)
    # print("l2 norm")
    # print(l2_norm)
    # print("inmatNormalised")
    # print(inmatNormalised)
    ##Test if all samples are normalised
    # testNorm = np.round(tf.reduce_sum(inmatNormalised ** 2, axis=1), decimals=4) == 1.0
    # print(testNorm)
    # if any(testNorm == False):
    #     print("Normalisation not performed on Batch dimension")

    SIMILARITY_MAT = tf.matmul(inmatNormalised, inmatNormalised, transpose_b=True)
    print("inmat normalised")
    print(SIMILARITY_MAT)
    SIMILARITY_MAT = tf.linalg.set_diag(SIMILARITY_MAT, diagonal=tf.zeros(batch_size, dtype='float32'))
    print("SIMILARITY_MAT", SIMILARITY_MAT)
    # if y_labels.shape[1] != no_classes:
    #     print("Number of classes mismatch ")
    #     sys.exit()

    # tf.matmul(y, SIMILARITY_MAT)
    # pos_label_mat = np.zeros(shape=(batch_size, batch_size))
    # print("pos_label_mat", pos_label_mat.shape)
    # for sample in range(batch_size):
    #     idx = y_labels[sample,:] == 1.0
    #     pos_label_mat[sample:,] = y_labels[:,idx]
    pos_label_mat = tf.matmul(y_labels, y_labels, transpose_b=True)
    pos_label_mat = tf.cast(pos_label_mat, tf.float32)

    neg_label_mat = 1.0 - pos_label_mat

    print("pos_label_mat")
    print(pos_label_mat)
    ######################Multiplying with alpha to get alpha *1 matrix
    # pos_label_mat = pos_label_mat

    # neg_label_mat = neg_label_mat
    ###Substract the lambda matrix
    # tf.reshape(x0, (k.shape(x0)[0], 1, k.shape(x0)[1]))

    lambda_1 = tf.constant([lambda_1])
    # multiples = tf.constant([k.shape(SIMILARITY_MAT)[0]])
    multiples = tf.constant([batch_size * batch_size])
    lambda_vec = tf.tile(lambda_1, multiples)
    print("lambda_vec")
    print(lambda_vec)
    # lambda_mat = tf.reshape(lambda_vec, (k.shape(inmat)[0], k.shape(inmat)[0]))
    lambda_mat = tf.reshape(lambda_vec, [batch_size, batch_size])
    lambda_mat = tf.cast(lambda_mat, tf.float32)
    print("lambda_mat")
    print(lambda_mat)
    ###Substract the lambda matrix
    SIMILARITY_MAT = SIMILARITY_MAT - lambda_mat

    ##The diagonal of this matrix would be the postive similarity
    # positive_similarity_mat = tf.matmul(pos_label_mat, SIMILARITY_MAT)
    positive_similarity_mat = SIMILARITY_MAT * -alpha
    # positive_similarity_mat = tf.matmul(pos_label_mat, SIMILARITY_MAT)
    # positive_similarity = tf.linalg.tensor_diag_part(positive_similarity_mat)
    # print("positive_similarity")
    # print(positive_similarity)

    ##The
    print("neg_label_mat")
    print(neg_label_mat)
    # negative_similarity_mat = tf.matmul(neg_label_mat, SIMILARITY_MAT)
    negative_similarity_mat = SIMILARITY_MAT * beta
    # negative_similarity_mat = tf.matmul(neg_label_mat, negative_similarity_mat)
    # negative_similarity = tf.linalg.tensor_diag_part(negative_similarity_mat)

    # postivie_exp = tf.math.exp(positive_similarity)
    # print("negative_similarity")
    # negative_exp = tf.math.exp(negative_similarity)
    # print(negative_exp)

    ##Exp
    positive_similarity_mat = tf.math.exp(positive_similarity_mat)
    negative_similarity_mat = tf.math.exp(negative_similarity_mat)

    print("positive_similarity_mat")
    print(positive_similarity_mat)

    ##Summation for entire batch
    positive_similarity_mat = tf.matmul(pos_label_mat, positive_similarity_mat)
    negative_similarity_mat = tf.matmul(neg_label_mat, negative_similarity_mat)

    ##
    positive_similarity = tf.linalg.tensor_diag_part(positive_similarity_mat)
    negative_similarity = tf.linalg.tensor_diag_part(negative_similarity_mat)

    print("positive_similarity")
    print(positive_similarity)

    print("negative_similarity")
    print(negative_similarity)

    ##Add 1
    positive_similarity = 1 + positive_similarity
    negative_similarity = 1 + negative_similarity

    ##log & scale
    positive_similarity = tf.math.log(positive_similarity) * (1. / alpha)
    negative_similarity = tf.math.log(negative_similarity) * (1. / beta)

    ##sum and mean
    loss = tf.reduce_sum((positive_similarity + negative_similarity)) * (1. / batch_size)
    # print(loss)
    return loss

    ##


# def dice_loss(smooth, thresh):
#   def dice(y_true, y_pred)
#     return -dice_coef(y_true, y_pred, smooth, thresh)
#   return dice

def cseloss(alpha, beta, batch_size=16):
    def lossmask(y_true, y_pred):
        return LossCse673(y_pred,  y_true, batch_size=batch_size, alpha=alpha, lambda_1=1, no_classes=98, beta=beta)
    return lossmask


# if __name__ == '__main__':
#     inmat = np.random.rand(2000, 512)
#     # Losscse673(inmat, 2)
#     y_label = [[0, 0, 1],
#                [0, 0, 1]]
#     y_label = np.array(y_label)
#     print(y_label.shape)
#     print(np.array(y_label))
#
#     predictions = tf.constant([[0.2, 0.4, 0.6],
#                                [0.1, 0.8, 0.1]])
#
#     # y_label = np.array([[0, 0, 1], [0, 0, 1]])
#     y_label = tf.Variable([[0, 0, 1], [0, 0, 1]], dtype='float64')
#
#     # LossCse673(predictions, 2, y_label, alpha=0.1, lambda_1=0.3, no_classes=3, beta=0.1)
#     loss = cseloss(alpha=0.1, beta=0.1)
#     print(loss(y_label, predictions))
