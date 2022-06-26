





# coding : utf-8
# from clogistic import LogisticRegression
from sklearn.datasets import load_breast_cancer

# from sklearn.linear_model import LogisticRegression
import numpy as np

    # from clogistic import LogisticRegression
from scipy.optimize import Bounds
import numpy as np
import sys
# sys.path.append('.')
from logistic_mini import LogisticRegression

def funL1():
    X,y = load_breast_cancer(return_X_y=True)
    clf = LogisticRegression(solver='lbfgs',penalty='l1')
    clf.fix(X,y)
    np.r_
    print('aa')

def fun2():

    # from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True)
    y = 1 - y

    lb = np.r_[np.zeros(X.shape[1]), -np.inf]
    # print('lb {}'.format(lb.shape))
    # score
    # 0.9578207381370826
    # intercept[-22.35343407]

    # lb = np.r_[np.zeros(X.shape[1]), 0]
    # score
    # 0.37258347978910367
    # intercept[-3.17464716e-09]

    # lb = np.r_[np.zeros(X.shape[1])]


    ub = np.r_[np.full(X.shape[1], np.inf), np.inf]

    # ub = np.r_[np.full(X.shape[1], np.inf)]

    bounds = Bounds(lb, ub)

    clf = LogisticRegression(solver="ecos", penalty="l1")
    clf.fit(X, y, bounds=bounds)

    print('score {}'.format(clf.score(X,y)))
    print('intercept {}'.format(clf.intercept_))

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

def fun5():

    from scipy.optimize import LinearConstraint
    X, y = load_breast_cancer(return_X_y=True)
    print('X {}'.format(X.shape))
    lb = np.zeros((X.shape[1]))
    # lb_ = np.zeros((X.shape[1]+1))
    # lb = np.r_[np.zeros(X.shape[1]-1),0]
    # ub = np.r_[np.full(X.shape[1]-1,np.inf), np.inf]
    ub = np.full(X.shape[1], np.inf)
    # ub_ = np.full(X.shape[1]+1,np.inf)
    # bounds = Bounds(lb_, ub_)
    print('lb {}'.format(lb.shape))
    print('un {}'.format(ub.shape))
    A = np.zeros((X.shape[1], X.shape[1] + 1))
    print('A {} '.format(A.shape))
    print('X.s {}'.format(X.shape[1]))
    for i in range(X.shape[1]):
        A[i, i:i + 2] = np.array([-1, 1])
        # print('A {}'.format(A))
    print('A {} {}'.format(A.shape, A[-1, -1]))

    constraints = LinearConstraint(A, lb, ub)

    clf = LogisticRegression(solver="lxw", penalty="l2")
    method = "Nelder-Mead"
    print('===================== {} ======='.format(method))
    clf.fit(X, y, constraints=constraints,method=method)

    print('{} score {}'.format(method,clf.score(X, y)))
    print('intercept {}'.format(clf.intercept_))
    print('coef_ {}'.format(clf.coef_))
    # print('res {} {}'.format(method,clf.res.x))

    method = "Newton-CG"
    print('===================== {} ======='.format(method))
    clf.fit(X, y, constraints=constraints, method=method)

    print('{} score {}'.format(method, clf.score(X, y)))
    print('intercept {}'.format(clf.intercept_))
    print('coef_ {}'.format(clf.coef_))
    print('res {} {}'.format(method, clf.res.success))
    # print('res {} {}'.format(method, clf.res.x))


    method = "trust-constr"
    print('===================== {} ======='.format(method))
    clf.fit(X, y, constraints=constraints, method=method)

    print('{} score {}'.format(method, clf.score(X, y)))
    print('intercept {}'.format(clf.intercept_))
    print('coef_ {}'.format(clf.coef_))
    print('res {} {}'.format(method, clf.res.success))
    # print('res {} {}'.format(method, clf.res.x))
    # trust - constr
    # score
    # 0.9209138840070299
    # intercept[15.30356143]
    # coef_[[-0.18272187 - 0.18272187 - 0.00501516 - 0.00501516 - 0.00501516 - 0.00501516
    #        - 0.00501516 - 0.00501516 - 0.00501516 - 0.00501516 - 0.00501516 - 0.00501516
    #        - 0.00501516 - 0.00501516 - 0.00501516 - 0.00501516 - 0.00501516 - 0.00501516
    #        - 0.00501516 - 0.00501516 - 0.00501516 - 0.00501516 - 0.00501516 - 0.00501516
    #        - 0.00501515 - 0.00501514 - 0.00501512 - 0.00501509 - 0.00501502 - 0.00501474]]
    # res
    # trust - constr
    # True
def fun7():
    from sklearn.linear_model import LogisticRegression
    X, y = load_breast_cancer(return_X_y=True)

    clf = LogisticRegression(solver="newton-cg",penalty="l2",max_iter=10000)

    method = clf.solver

    print('===================== {} ======='.format(method))
    print('penalty: {}'.format(clf.penalty))
    print('max_iter: {}'.format(clf.max_iter))
    # clf.fit(X, y, method=method,bounds=bounds)
    clf.fit(X, y)


    print('score: {}'.format(clf.score(X, y)))
    print('intercept: {}'.format(clf.intercept_))
    print('coef_: {}'.format(clf.coef_))
    # print('res {} {}'.format(method, clf.res.success))

    # == == == == == == == == == == = lbfgs == == == =
    # penalty: none
    # max_iter: 10000
    # score: 0.9876977152899824
    # intercept: [34.02984597]
    # coef_: [[3.31445364e+00 - 1.20533934e-01  4.46757143e-01 - 4.90070847e-02
    #          - 2.17163624e+01  9.57167462e+00 - 2.11397964e+01 - 4.04943418e+01
    #          - 7.72839658e-01  4.45657354e+00  1.21811568e+00  2.66588562e+00
    #          1.41986482e+00 - 3.74405072e-01 - 4.44753589e+00  3.95539480e+01
    #          4.88910217e+01 - 2.58956189e+00  9.83494400e+00  7.19527597e+00
    #          - 2.55335547e+00 - 5.46869475e-01 - 2.09318208e-01  1.63183507e-02
    #          - 4.68846895e+01  1.37821522e+01 - 1.42019184e+01 - 6.90567978e+01
    #          - 1.15324127e+01  3.49230633e+00]]

    # == == == == == == == == == == = lbfgs == == == =
    # penalty: none
    # max_iter: 11000
    # score: 0.9876977152899824
    # intercept: [34.02984597]
    # coef_: [[3.31445364e+00 - 1.20533934e-01  4.46757143e-01 - 4.90070847e-02
    #          - 2.17163624e+01  9.57167462e+00 - 2.11397964e+01 - 4.04943418e+01
    #          - 7.72839658e-01  4.45657354e+00  1.21811568e+00  2.66588562e+00
    #          1.41986482e+00 - 3.74405072e-01 - 4.44753589e+00  3.95539480e+01
    #          4.88910217e+01 - 2.58956189e+00  9.83494400e+00  7.19527597e+00
    #          - 2.55335547e+00 - 5.46869475e-01 - 2.09318208e-01  1.63183507e-02
    #          - 4.68846895e+01  1.37821522e+01 - 1.42019184e+01 - 6.90567978e+01
    #          - 1.15324127e+01  3.49230633e+00]]

    # == == == == == == == == == == = saga == == == =
    # penalty: none
    # max_iter: 10000
    # score: 0.9226713532513181
    # intercept: [0.00233021]
    # coef_: [[1.62368270e-02  2.35054808e-03  8.06538948e-02  1.56742529e-02
    #          7.64210624e-05 - 4.60430525e-04 - 8.33093551e-04 - 3.33181484e-04
    #          1.59680617e-04  8.84220224e-05  1.39095915e-04  5.00591431e-04
    #          - 1.66372707e-03 - 2.83959443e-02 - 5.68898022e-07 - 1.29459139e-04
    #          - 1.73412701e-04 - 3.86499011e-05 - 5.92382583e-07 - 8.07131449e-06
    #          1.70147199e-02 - 4.89865155e-03  6.70087917e-02 - 2.92748635e-02
    #          4.14073098e-05 - 1.71612005e-03 - 2.33702065e-03 - 6.02537428e-04
    #          - 5.26916421e-06 - 3.22149453e-05]]

    # == == == == == == == == == == = saga == == == =
    # penalty: none
    # max_iter: 8000
    # score: 0.9226713532513181
    # intercept: [0.00232991]
    # coef_: [[1.62351806e-02  2.34472550e-03  8.06520629e-02  1.56740638e-02
    #          7.64601057e-05 - 4.60198900e-04 - 8.32764195e-04 - 3.33024477e-04
    #          1.59684570e-04  8.84248044e-05  1.39221267e-04  5.00044737e-04
    #          - 1.66290972e-03 - 2.83903101e-02 - 5.69942525e-07 - 1.29429515e-04
    #          - 1.73352502e-04 - 3.86333847e-05 - 5.82145069e-07 - 8.06908441e-06
    #          1.70136300e-02 - 4.89912659e-03  6.70100397e-02 - 2.92746571e-02
    #          4.14632251e-05 - 1.71549706e-03 - 2.33630010e-03 - 6.02249191e-04
    #          - 5.10260193e-06 - 3.21906219e-05]]

    # == == == == == == == == == == = sag == == == =
    # penalty: none
    # max_iter: 8000
    # score: 0.9226713532513181
    # intercept: [0.00366985]
    # coef_: [[2.49011900e-02 - 1.30078866e-02  1.07852778e-01  1.35424533e-02
    #          1.21634353e-05 - 1.17919739e-03 - 1.91733070e-03 - 7.56617368e-04
    #          6.43666466e-05  8.73358768e-05  3.06487257e-04  2.81645302e-04
    #          - 3.89642082e-03 - 3.57338899e-02 - 1.34774130e-05 - 3.24456384e-04
    #          - 4.29347967e-04 - 9.83790331e-05 - 3.54921348e-05 - 2.34519711e-05
    #          2.60133872e-02 - 3.59508942e-02  6.96243527e-02 - 2.97514846e-02
    #          - 1.38042660e-04 - 4.29670497e-03 - 5.60236019e-03 - 1.46443948e-03
    #          - 5.20738726e-04 - 2.25991488e-04]]

    # == == == == == == == == == == = newton - cg == == == =
    # penalty: l2
    # max_iter: 10000
    # score: 0.9578207381370826
    # intercept: [28.08899349]
    # coef_: [[1.01456382  0.18138255 - 0.27569729  0.0226507 - 0.17839758 - 0.22084002
    #          - 0.53505267 - 0.29511993 - 0.26624411 - 0.03025814 - 0.07839993  1.263851
    #          0.11658972 - 0.10881538 - 0.02509762  0.06720758 - 0.0360128 - 0.03799299
    #          - 0.0367822   0.01398789  0.13786713 - 0.43764205 - 0.10580434 - 0.01363256
    #          - 0.35635454 - 0.68786615 - 1.42190555 - 0.60235892 - 0.73091405 - 0.0950031]]


def Non_negative_cons():
    from scipy.optimize import LinearConstraint
    X, y = load_breast_cancer(return_X_y=True)
    y[y==0] = -1
    lb = np.zeros((X.shape[1]))
    lb_ = np.zeros((X.shape[1]+1))

    ub = np.full(X.shape[1], np.inf)
    ub_ = np.full(X.shape[1]+1, np.inf)

    bounds = Bounds(lb_, ub_)
    print('lb {}'.format(lb.shape))
    print('un {}'.format(ub.shape))

    A = np.full((X.shape[1], X.shape[1]+1),0)
    print('A {}'.format(type(A)))
    print('A {} '.format(A.shape))
    print('X.s {}'.format(X.shape[1]))
    for i in range(X.shape[1]):
        A[i,i] = 1
        # print('A {}'.format(A))
    print('A {} {}'.format(A.shape, A[-1, -1]))

    constraints = LinearConstraint(A, lb, ub)
    # clf = LogisticRegression(solver="lxw", penalty="none")
    # method = "Nelder-Mead"
    # print('===================== {} ======='.format(method))
    # clf.fit(X, y,bounds=bounds,method=method)
    #
    # print('{} score {}'.format(method,clf.score(X, y)))
    # print('intercept {}'.format(clf.intercept_))
    # print('coef_ {}'.format(clf.coef_))
    # # print('res {} {}'.format(method,clf.res.x))
    #
    # method = "Newton-CG"
    # print('===================== {} ======='.format(method))
    # clf.fit(X, y,bounds=bounds,method=method)
    #
    # print('{} score {}'.format(method, clf.score(X, y)))
    # print('intercept {}'.format(clf.intercept_))
    # print('coef_ {}'.format(clf.coef_))
    # print('res {} {}'.format(method, clf.res.success))
    # # print('res {} {}'.format(method, clf.res.x))

    #
    # method = "trust-constr"
    # print('===================== {} ======='.format(method))
    # clf.fit(X, y,method=method,constraints=constraints)
    #
    # print('{} score {}'.format(method, clf.score(X, y)))
    # print('intercept {}'.format(clf.intercept_))
    # print('coef_ {}'.format(clf.coef_))
    # print('res {} {}'.format(method, clf.res.success))
    # print('res {} {}'.format(method, clf.res.x))
    # y==-1
# L2 正则
# ===================== trust-constr =======
# trust-constr score 0.6274165202108963
# intercept [0.48239057]
# coef_ [[1.35802952e-11 1.93954466e-11 1.94877677e-12 2.27477920e-13
#   4.27932882e-09 1.10474108e-09 6.26834893e-10 1.40957283e-09
#   4.83991964e-09 2.21198612e-02 2.21445103e-10 3.03164007e-02
#   3.09431310e-11 1.41449719e-12 5.25654438e-02 4.33361973e-10
#   5.38493798e-09 1.76001831e-08 7.09296450e-03 1.64207951e-07
#   9.31409740e-12 1.23439172e-11 1.34030098e-12 1.35863256e-13
#   3.43298611e-09 3.76007637e-10 2.53635875e-10 6.69776784e-10
#   1.36223163e-09 7.48485484e-09]]
# res trust-constr True
#
# ===================== trust-constr =======
# trust-constr score 0.6274165202108963
# intercept [0.48239057]
# coef_ [[1.35802952e-11 1.93954466e-11 1.94877677e-12 2.27477920e-13
#   4.27932882e-09 1.10474108e-09 6.26834893e-10 1.40957283e-09
#   4.83991964e-09 2.21198612e-02 2.21445103e-10 3.03164007e-02
#   3.09431310e-11 1.41449719e-12 5.25654438e-02 4.33361973e-10
#   5.38493798e-09 1.76001831e-08 7.09296450e-03 1.64207951e-07
#   9.31409740e-12 1.23439172e-11 1.34030098e-12 1.35863256e-13
#   3.43298611e-09 3.76007637e-10 2.53635875e-10 6.69776784e-10
#   1.36223163e-09 7.48485484e-09]]
# res trust-constr True

# ===================== trust-constr =======
# trust-constr score 0.6274165202108963
# intercept [0.17889701]
# coef_ [[6.88761583e-11 9.86666265e-11 9.56717533e-12 8.27203477e-13
#   3.13376498e-08 9.29275633e-09 4.92886932e-09 1.13882013e-08
#   1.65018761e-08 1.64670115e-06 8.11847988e-10 1.66153612e-08
#   1.48593144e-10 6.70152662e-12 4.90536334e+01 5.74619752e-08
#   2.30209849e-08 6.56837793e-08 1.77989622e-06 6.36760569e-07
#   4.52216816e-11 6.39388695e-11 6.56299634e-12 4.10800274e-13
#   1.63044398e-08 1.68603113e-09 1.09938529e-09 3.47942709e-09
#   6.92924388e-09 3.31145267e-08]]
# res trust-constr True
    ############
    ####  method use L-BFGS-B and TNC
    ############
    method = "L-BFGS-B"

    # clf = LogisticRegression(solver="lxw", penalty="none",max_iter=150)
    # fix_initial 初值赋值更改，验证参数收敛是否一致
    clf = LogisticRegression(solver="lxw", penalty="none",max_iter=60,fix_initial=True)

    clf.fit(X, y, method=method,bounds=bounds)
    print('===================== {} ==========='.format(method))
    print('max_iter : {}'.format(clf.max_iter))
    print('penalty : {}'.format(clf.penalty))

    print('score : {}'.format(clf.score(X, y)))
    print('intercept : {}'.format(clf.intercept_))
    print('coef_  : {}'.format(clf.coef_))
    print('res: {}  optimizer : {}'.format(method, clf.res.success))

    # == == == == == == == == == == = L - BFGS - B == == == == == =
    # max_iter: 30
    # penalty: none
    # score: 0.6274165202108963
    # intercept: [0.00667868]
    # coef_: [[0.00000000e+00 1.58544953e-02 0.00000000e+00 0.00000000e+00
    #          3.41957922e-04 0.00000000e+00 0.00000000e+00 0.00000000e+00
    #          6.65275978e-04 4.26636563e-04 0.00000000e+00 8.09975829e-03
    #          0.00000000e+00 0.00000000e+00 5.95576517e-05 0.00000000e+00
    #          0.00000000e+00 0.00000000e+00 1.41356898e-04 1.30285931e-05
    #          0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
    #          3.04743751e-04 0.00000000e+00 0.00000000e+00 0.00000000e+00
    #          3.83613404e-04 2.07923234e-04]]
    # res
    # L - BFGS - B
    # False

    # == == == == == == == == == == = L - BFGS - B == == == == == =
    # max_iter: 50
    # penalty: none
    # score: 0.6274165202108963
    # intercept: [0.1967258]
    # coef_: [[0.         0.         0.         0.         0.00889884 0.
    #          0.         0.         0.0174621  0.01259587 0.         0.23684945
    #          0.         0.         0.00180309 0.         0.         0.
    #          0.00418088 0.00033521 0.         0.         0.         0.
    #          0.00669678 0.         0.         0.         0.00518067 0.00472869]]
    # res: L - BFGS - B
    # optimizer: True

    # == == == == == == == == == == = L - BFGS - B == == == == == =
    # max_iter: 50
    # penalty: l2
    # score: 0.6274165202108963
    # intercept: [0.19640032]
    # coef_: [[0.         0.         0.         0.         0.00888393 0.
    #          0.         0.         0.01743283 0.01257477 0.         0.23645251
    #          0.         0.         0.00180007 0.         0.         0.
    #          0.00417387 0.00033464 0.         0.         0.         0.
    #          0.00668555 0.         0.         0.         0.00517197 0.00472076]]
    # res: L - BFGS - B
    # optimizer: True

# ===================== L-BFGS-B ===========
# max_iter : 60
# penalty : none
# score : 0.6274165202108963
# intercept : [0.19999506]
# coef_  : [[0.         0.         0.         0.         0.         0.
#   0.         0.         0.         0.0020697  0.         0.24097333
#   0.         0.         0.00400276 0.         0.         0.
#   0.00099565 0.01133466 0.         0.         0.         0.
#   0.         0.         0.         0.         0.         0.        ]]
# res: L-BFGS-B  optimizer : True




# ===================== TNC ===========
# max_iter : 40
# penalty : none
# score : 0.6274165202108963
# intercept : [0.13340021]
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
# intercept : [0.17889717]
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
# intercept : [0.17889717]
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
# intercept : [0.48239063]
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
# intercept : [0.17889717]
# coef_  : [[ 0.          0.          0.          0.          0.          0.
#    0.          0.          0.          0.          0.          0.
#    0.          0.         49.05363767  0.          0.          0.
#    0.          0.          0.          0.          0.          0.
#    0.          0.          0.          0.          0.          0.        ]]
# res: TNC  optimizer : True




if __name__ == '__main__':
    Non_negative_cons()
    # X,y = load_breast_cancer(return_X_y=True)
    # print('X,y {} {}'.format(X.shape,y.shape))
    # print(X,y)
