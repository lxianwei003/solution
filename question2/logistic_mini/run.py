





# coding : utf-8
from sklearn.datasets import load_breast_cancer
from scipy.optimize import LinearConstraint

import numpy as np
from scipy.optimize import Bounds
import numpy as np
from logistic_mini import LogisticRegression


def fun3():

    from scipy.optimize import LinearConstraint
    X, y = load_breast_cancer(return_X_y=True)
    print('X {}'.format(X.shape))
    lb = np.zeros((X.shape[1]))
    # lb = np.r_[np.zeros(X.shape[1]-1),0]
    # ub = np.r_[np.full(X.shape[1]-1,np.inf), np.inf]
    ub = np.full(X.shape[1],np.inf)

    bounds = Bounds(lb, ub)
    print('lb {}'.format(lb.shape))
    print('un {}'.format(ub.shape))
    A = np.zeros((X.shape[1],X.shape[1]+1))
    print('A {} '.format(A.shape))
    print('X.s {}'.format(X.shape[1]))
    for i in range(X.shape[1]):
        A[i,i:i+2] = np.array([-1,1])
        # print('A {}'.format(A))
    print('A {} {}'.format(A.shape,A[-1,-1]))

    constraints = LinearConstraint(A,lb,ub)


    clf = LogisticRegression(solver="ecos", penalty="elasticnet",l1_ratio=0.5)
    clf.fit(X, y,constraints=constraints)

    print('score {}'.format(clf.score(X, y)))
    print('intercept {}'.format(clf.intercept_))
    print('coef_ {}'.format(clf.coef_))
    # coef_[[-0.18136544 - 0.18136544 - 0.00501176 - 0.00501176 - 0.00501177 - 0.00501177
    #        - 0.00501178 - 0.00501178 - 0.00501179 - 0.00501179 - 0.00501179 - 0.0050118
    #        - 0.0050118 - 0.0050118 - 0.00501181 - 0.00501181 - 0.00501182 - 0.00501182
    #        - 0.00501183 - 0.00501183 - 0.00501184 - 0.00501184 - 0.00501184 - 0.00501184
    #        - 0.00501185 - 0.00501185 - 0.00501185 - 0.00501184 - 0.00501183 - 0.00501173]]
    # #
    # X(569, 30)
    # lb(30, )
    # un(30, )
    # A(30, 31)
    # X.s
    # 30
    # A(30, 31)
    # 1.0
    # score
    # 0.9209138840070299
    # intercept[15.251803]

    clf = LogisticRegression(solver="ecos", penalty="l1")
    clf.fit(X, y, constraints=constraints)

    print('score {}'.format(clf.score(X, y)))
    print('intercept {}'.format(clf.intercept_))
    print('coef_ {}'.format(clf.coef_))
    # score
    # 0.9191564147627417
    # intercept[15.20015283]
    # coef_[[-0.18000882 - 0.18000882 - 0.00500848 - 0.00500848 - 0.00500849 - 0.00500849
    #        - 0.00500849 - 0.0050085 - 0.0050085 - 0.00500851 - 0.00500851 - 0.00500851
    #        - 0.00500852 - 0.00500852 - 0.00500852 - 0.00500853 - 0.00500853 - 0.00500854
    #        - 0.00500854 - 0.00500854 - 0.00500855 - 0.00500855 - 0.00500855 - 0.00500855
    #        - 0.00500855 - 0.00500856 - 0.00500856 - 0.00500855 - 0.00500853 - 0.00500795]]

    clf = LogisticRegression(solver="ecos", penalty="l2")
    clf.fit(X, y, constraints=constraints)

    print('score {}'.format(clf.score(X, y)))
    print('intercept {}'.format(clf.intercept_))
    print('coef_ {}'.format(clf.coef_))
    # score
    # 0.9209138840070299
    # intercept[15.30357001]
    # coef_[[-0.18272197 - 0.18272197 - 0.00501515 - 0.00501515 - 0.00501515 - 0.00501515
    #        - 0.00501515 - 0.00501515 - 0.00501516 - 0.00501516 - 0.00501516 - 0.00501516
    #        - 0.00501516 - 0.00501516 - 0.00501516 - 0.00501516 - 0.00501517 - 0.00501517
    #        - 0.00501517 - 0.00501517 - 0.00501517 - 0.00501517 - 0.00501517 - 0.00501517
    #        - 0.00501517 - 0.00501518 - 0.00501518 - 0.00501517 - 0.00501517 - 0.00501515]]
def fun4():
    from clogistic import LogisticRegression
    from scipy.optimize import LinearConstraint
    X,y= load_breast_cancer(return_X_y=True)
    lb = np.array([0.0])
    ub = np.array([0.5])
    a = np.array([[-1,2]])
    print('a {}'.format(a.shape))
    print('lb {}'.format(lb.shape))
    print('ub {}'.format(ub.shape))

    A = np.zeros((1,X.shape[1]+1))
    A[0,:2] = np.array([-1,1])

    print('A {}'.format(A.shape))
    constraints = LinearConstraint(A,lb,ub)
    clf = LogisticRegression(solver='ecos',penalty='elasticnet',l1_ratio=0.5)
    clf.fit(X,y,constraints=constraints)
    print('coef_ {}'.format(clf.coef_))
    print('clf.intercept_ {}'.format(clf.intercept_))

    # a(1, 2)
    # lb(1, )
    # ub(1, )
    # A(1, 31)
    # coef_[[-2.60422355e-01  2.39577652e-01 - 1.92012495e-01  2.90666809e-02
    #        - 3.97370760e-08 - 7.16334693e-08 - 2.33698014e-01 - 9.86651029e-08
    #        - 7.46276071e-08 - 7.02667735e-09 - 1.55098920e-08  1.42995024e+00
    #        4.09734138e-08 - 1.08592016e-01 - 4.97863432e-09  8.06378661e-09
    #        - 1.08950790e-08 - 7.51201540e-09 - 8.62403097e-09  1.75870056e-09
    #        8.35103279e-08 - 4.84664844e-01 - 1.00378366e-01 - 1.14688678e-02
    #        - 1.56122189e-07 - 5.95038188e-01 - 2.09885806e+00 - 2.82823803e-01
    #        - 5.77869754e-01 - 2.10321102e-08]]
    # clf.intercept_[34.3868653]


def logistic_reg():
    print("******************** logistic_reg ********************")
    from sklearn.linear_model import LogisticRegression
    X, y = load_breast_cancer(return_X_y=True)

    clf = LogisticRegression(solver="lbfgs",penalty="none",max_iter=11000)

    method = clf.solver

    clf.fit(X, y)
    print('===================== {} ==========='.format(method))
    print('max_iter : {}'.format(clf.max_iter))
    print('penalty : {}'.format(clf.penalty))

    print('score : {}'.format(clf.score(X, y)))
    print('coef_  : {}'.format(clf.coef_))

    # == == == == == == == == == == = lbfgs == == == == == =
    # max_iter: 10000
    # penalty: none
    # score: 0.9876977152899824
    # coef_: [[3.31445364e+00 - 1.20533934e-01  4.46757143e-01 - 4.90070847e-02
    #          - 2.17163624e+01  9.57167462e+00 - 2.11397964e+01 - 4.04943418e+01
    #          - 7.72839658e-01  4.45657354e+00  1.21811568e+00  2.66588562e+00
    #          1.41986482e+00 - 3.74405072e-01 - 4.44753589e+00  3.95539480e+01
    #          4.88910217e+01 - 2.58956189e+00  9.83494400e+00  7.19527597e+00
    #          - 2.55335547e+00 - 5.46869475e-01 - 2.09318208e-01  1.63183507e-02
    #          - 4.68846895e+01  1.37821522e+01 - 1.42019184e+01 - 6.90567978e+01
    #          - 1.15324127e+01  3.49230633e+00]]
    ### 迭代次数在10000 再增加迭代，参数无变化，全局最优
    # == == == == == == == == == == = lbfgs == == == == == =
    # max_iter: 11000
    # penalty: none
    # score: 0.9876977152899824
    # coef_: [[3.31445364e+00 - 1.20533934e-01  4.46757143e-01 - 4.90070847e-02
    #          - 2.17163624e+01  9.57167462e+00 - 2.11397964e+01 - 4.04943418e+01
    #          - 7.72839658e-01  4.45657354e+00  1.21811568e+00  2.66588562e+00
    #          1.41986482e+00 - 3.74405072e-01 - 4.44753589e+00  3.95539480e+01
    #          4.88910217e+01 - 2.58956189e+00  9.83494400e+00  7.19527597e+00
    #          - 2.55335547e+00 - 5.46869475e-01 - 2.09318208e-01  1.63183507e-02
    #          - 4.68846895e+01  1.37821522e+01 - 1.42019184e+01 - 6.90567978e+01
    #          - 1.15324127e+01  3.49230633e+00]]

def Non_negative_cons():

    print("******************** Non_negative_cons********************")

    X, y = load_breast_cancer(return_X_y=True)
    # sklearn y = {-1,1}
    y[y == 0] = -1
    lb_ = np.zeros((X.shape[1] + 1))

    ub_ = np.full(X.shape[1] + 1, np.inf)

    bounds = Bounds(lb_, ub_)
    print('lb_ {}'.format(lb_.shape))
    print('ub_ {}'.format(ub_.shape))

    ############
    ####  method 可选 L-BFGS-B and TNC
    ####  solver='lxw' 调用 _fix_lxw 执行minimize
    ####  max_iter 迭代次数
    #### fix_initial 是否更改初始值
    #### bounds 设置 constraints 边界，问题1 即 a1>=0,a2>=2,a3>=3....

    #   return param
    #   max_iter ,penalty（none 即没用正则）
    #   score 分类准确率
    #   coef_ coefficient
    #
    ############
    method = "L-BFGS-B"

    clf = LogisticRegression(solver="lxw", penalty="none", max_iter=150)
    # fix_initial 初值赋值更改，验证参数收敛是否一致，更改
    #     clf = LogisticRegression(solver="lxw", penalty="none",max_iter=60,fix_initial=True)

    clf.fit(X, y, method=method, bounds=bounds)
    print('===================== {} ==========='.format(method))
    print('max_iter : {}'.format(clf.max_iter))
    print('penalty : {}'.format(clf.penalty))

    print('score : {}'.format(clf.score(X, y)))
    print('coef_  : {}'.format(clf.coef_))
    print('res: {}  optimizer : {}'.format(method, clf.res.success))

    # L-BFGS-B 迭代 50、60次，参数相同，有最优解，
    # == == == == == == == == == == = L - BFGS - B == == == == == =
    # max_iter: 50
    # penalty: none
    # score: 0.6274165202108963
    # coef_: [[0.         0.         0.         0.         0.00889884 0.
    #          0.         0.         0.0174621  0.01259587 0.         0.23684945
    #          0.         0.         0.00180309 0.         0.         0.
    #          0.00418088 0.00033521 0.         0.         0.         0.
    #          0.00669678 0.         0.         0.         0.00518067 0.00472869]]
    # res: L - BFGS - B
    # optimizer: True

    # ===================== L-BFGS-B ===========
    # max_iter : 60
    # penalty : none
    # score : 0.6274165202108963
    # coef_  : [[0.         0.         0.         0.         0.         0.
    #   0.         0.         0.         0.0020697  0.         0.24097333
    #   0.         0.         0.00400276 0.         0.         0.
    #   0.00099565 0.01133466 0.         0.         0.         0.
    #   0.         0.         0.         0.         0.         0.        ]]
    # res: L-BFGS-B  optimizer : True

    # 加入 l2的结果
    # == == == == == == == == == == = L - BFGS - B == == == == == =
    # max_iter: 50
    # penalty: l2
    # score: 0.6274165202108963
    # coef_: [[0.         0.         0.         0.         0.00888393 0.
    #          0.         0.         0.01743283 0.01257477 0.         0.23645251
    #          0.         0.         0.00180007 0.         0.         0.
    #          0.00417387 0.00033464 0.         0.         0.         0.
    #          0.00668555 0.         0.         0.         0.00517197 0.00472076]]
    # res: L - BFGS - B
    # optimizer: True

    # ===================== TNC ===========
    # max_iter : 40
    # penalty : none
    # score : 0.6274165202108963
    # coef_  : [[ 0.          0.          0.          0.          0.          0.
    #    0.          0.          0.          1.1488147   0.          0.
    #    0.          0.         45.21328606  0.          0.          0.
    #    0.          0.          0.          0.          0.          0.
    #    0.          0.          0.          0.          0.          0.        ]]
    # res: TNC  optimizer : False

    #####  =========== 迭代50次以后优化器退出，为true，coef_稳定全局解
    # ===================== TNC ===========
    # max_iter : 50
    # penalty : none
    # score : 0.6274165202108963
    # coef_  : [[ 0.         0.         0.         0.         0.         0.
    #    0.         0.         0.         0.         0.         0.
    #    0.         0.        49.0536372  0.         0.         0.
    #    0.         0.         0.         0.         0.         0.
    #    0.         0.         0.         0.         0.         0.       ]]
    # res: TNC  optimizer : True

    # ===================== TNC ===========
    # max_iter : 60
    # penalty : none
    # score : 0.6274165202108963
    # coef_  : [[ 0.         0.         0.         0.         0.         0.
    #    0.         0.         0.         0.         0.         0.
    #    0.         0.        49.0536372  0.         0.         0.
    #    0.         0.         0.         0.         0.         0.
    #    0.         0.         0.         0.         0.         0.       ]]
    # res: TNC  optimizer : True

    ##### ====== 加入L2正则 迭代70次 收敛
    # ===================== TNC ===========
    # max_iter : 70
    # penalty : l2
    # score : 0.6274165202108963
    # coef_  : [[0.         0.         0.         0.         0.         0.
    #   0.         0.         0.         0.02211936 0.         0.0303164
    #   0.         0.         0.05256518 0.         0.         0.
    #   0.0070915  0.         0.         0.         0.         0.
    #   0.         0.         0.         0.         0.         0.        ]]
    # res: TNC  optimizer : True

    ### 更改赋初值，coef_收敛相同值，迭代次数增加到150，LR 加入 Non-negative constraint 使用TNC 有全局最优解
    # ===================== TNC ===========
    # max_iter : 150
    # penalty : none
    # score : 0.6274165202108963
    # coef_  : [[ 0.          0.          0.          0.          0.          0.
    #    0.          0.          0.          0.          0.          0.
    #    0.          0.         49.05363767  0.          0.          0.
    #    0.          0.          0.          0.          0.          0.
    #    0.          0.          0.          0.          0.          0.        ]]
    # res: TNC  optimizer : True
def order_cons():
    print('******************** order_cons ********************')
    X, y = load_breast_cancer(return_X_y=True)

    lb = np.zeros((X.shape[1]))
    ub = np.full(X.shape[1], np.inf)

    # print('lb {}'.format(lb.shape))
    # print('un {}'.format(ub.shape))
    #
    A = np.zeros((X.shape[1], X.shape[1] + 1))
    for i in range(X.shape[1]):
        A[i, i:i + 2] = np.array([1, -1])

    # print('A {} {}'.format(A.shape, A[-1, -1]))

    constraints = LinearConstraint(A, lb, ub)

    clf = LogisticRegression(solver="lxw", penalty="none",max_iter=225)
    #      fix_initial 初值赋值更改，验证参数收敛是否一致
    # clf = LogisticRegression(solver="lxw", penalty="none",max_iter=225,fix_initial=True)
    ############
    ####  method trust-constr
    ####  solver='lxw' 调用 _fix_lxw 执行minimize
    ####  max_iter 迭代次数
    #### fix_initial 是否更改初始值
    #### onstraints 为 即 a1>=a2>=a3 ....

    #   return param
    #   max_iter ,penalty（none 即没用正则）
    #   score 分类准确率
    #   coef_ coefficient
    #
    ############
    method = "trust-constr"

    clf.fit(X, y, constraints=constraints, method=method)

    print('===================== {} ==========='.format(method))
    print('max_iter : {}'.format(clf.max_iter))
    print('penalty : {}'.format(clf.penalty))

    print('score : {}'.format(clf.score(X, y)))
    print('coef_  : {}'.format(clf.coef_))
    print('res: {}  optimizer : {}'.format(method, clf.res.success))

    # == == == == == == == == == == = trust - constr == == == == == =
    # max_iter: 225
    # penalty: none
    # score: 0.961335676625659
    # coef_: [[3.80424902 - 0.02360999 - 0.02360999 - 0.02360999 - 0.02360999 - 0.02360999
    #          - 0.02360999 - 0.02360999 - 0.02360999 - 0.02360999 - 0.02360999 - 0.02360999
    #          - 0.02360999 - 0.02360999 - 0.02360999 - 0.02360999 - 0.02360999 - 0.02360999
    #          - 0.02360999 - 0.02360999 - 0.02360999 - 0.02360999 - 0.02360999 - 0.02360999
    #          - 2.25030929 - 2.25030933 - 2.25030944 - 7.35546295 - 7.35546298 - 7.35546301]]
    # res: trust - constr
    # optimizer: True

    ## 更改初值 参数与上个初值的 参数一致，目标函数全局最优解
    # == == == == == == == == == == = trust - constr == == == == == =
    # max_iter: 225
    # penalty: none
    # score: 0.961335676625659
    # coef_: [[3.80424876 - 0.02360998 - 0.02360998 - 0.02360998 - 0.02360998 - 0.02360998
    #          - 0.02360998 - 0.02360998 - 0.02360998 - 0.02360998 - 0.02360999 - 0.02360999
    #          - 0.02360999 - 0.02360999 - 0.02360999 - 0.02360999 - 0.02360999 - 0.02360999
    #          - 0.02360999 - 0.02360999 - 0.02360999 - 0.02360999 - 0.02360999 - 0.02360999
    #          - 2.25030933 - 2.25030948 - 2.25030978 - 7.35546156 - 7.35546174 - 7.35546193]]
    # res: trust - constr
    # optimizer: True

    ## 引入 正则 l2
    # == == == == == == == == == == = trust - constr == == == == == =
    # max_iter: 225
    # penalty: l2
    # score: 0.9490333919156415
    # coef_: [[2.04013969 - 0.00463975 - 0.00463975 - 0.00463975 - 0.0221412 - 0.02214125
    #          - 0.02214127 - 0.02214127 - 0.02214127 - 0.02214128 - 0.02214128 - 0.02214128
    #          - 0.02214129 - 0.02214129 - 0.02214129 - 0.02214129 - 0.02214129 - 0.02214129
    #          - 0.02214129 - 0.02214129 - 0.02214129 - 0.02214129 - 0.02214129 - 0.02214129
    #          - 0.48456143 - 1.484345 - 1.48434501 - 1.48434502 - 1.48434502 - 1.48434503]]
    # res: trust - constr
    # optimizer: True



if __name__ == '__main__':
    #  question1
    Non_negative_cons()

    # question2
    order_cons()

    # 更改迭代次数，验证 LR is  a global optimization algorithm
    logistic_reg()

