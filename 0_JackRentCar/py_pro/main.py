import numpy as np
import matplotlib.pyplot as plt

isPoisson = False


def getCost(a: int, b: int, move: int):
    # 计算在某状态时 给搬运值时的实际搬运情况
    lift_num = 0
    a_b = 0
    b_a = 0
    if move > 0:
        if move <= a:
            lift_num = lift_num + move
            a_b = move
        else:
            lift_num = lift_num + a
            a_b = a
    elif move == 0:
        a_b = 0
        b_a = 0
        lift_num = 0
    else:
        move = 0 - move
        if move <= b:
            lift_num = lift_num + move
            b_a = move
        else:
            lift_num = lift_num + b
            b_a = b
    after_a = a - a_b + b_a
    after_b = b + a_b - b_a
    return lift_num, after_a, after_b


def getProfit(after_a, after_b, rent_a, rent_b):
    total_rent = 0
    if after_a <= rent_a:
        total_rent = total_rent + after_a
        aafter_a = 0
    else:
        total_rent = total_rent + rent_a
        aafter_a = after_a - rent_a

    if after_b <= rent_b:
        total_rent = total_rent + after_b
        aafter_b = 0
    else:
        total_rent = total_rent + rent_b
        aafter_b = after_b - rent_b

    profit = total_rent * 10

    return aafter_a, aafter_b, profit


def getPoissonRate(para_lamda):
    samples_poi = np.random.poisson(para_lamda, size=100000)
    key_arr = np.zeros(15, dtype=float)
    for i in samples_poi:
        if i > 12:
            continue
        key_arr[i] += 1
    key_arr = key_arr / 100000.0
    return key_arr


def getStateValue(numA, numB, state_value_arr, best_action,
                  len_A_pro_list, len_B_pro_list,
                  re_A_pro_list, re_B_pro_list):
    state_value = 0.
    # 这个问题里的租车与还车 造就了reward的不确定性，应该在做完决策
    # 公式 V(s)= sigma_a: pi(a|s) * {sigma_s': p(s', r'|s, a) * [r(s, a, s') + discount * V(s’)]}
    # 由于我们只做贪心，sigma省略，若需要软策略，在此设置概率调整 best_action的取值即可
    # 一次迭代: 1.搬车reward 2.租车reward 3.还车 4.此时的numA numB才被决定，计算 discount * V（s'）
    # 可在此加入软策略 对 action 取值进行修改
    lift_num, after_a, after_b = getCost(numA, numB, best_action)
    state_value = state_value - lift_num * 2
    # 求
    for len_A in range(13):
        # print("round %d" % len_A)
        for len_B in range(13):
            posibility_len = len_A_pro_list[len_A] * len_B_pro_list[len_B]
            # 租车 服从泊松分布 这里依概率进行抽样
            aafter_a, aafter_b, profit = getProfit(after_a, after_b, len_A, len_B)
            # state_value = state_value + profit
            # 还车 服从泊松分布 这里依概率进行抽样
            # 直接使用 3，2 还车数量
            if isPoisson:
                for re_A in range(13):
                    for re_B in range(13):
                        posibility_re = re_A_pro_list[re_A] * re_B_pro_list[re_B]

                        aaafter_a = min(20, re_A + aafter_a)
                        aaafter_b = min(20, re_B + aafter_b)

                        # 计算V(s')
                        state_value += (posibility_len * posibility_re *
                                        (profit + 0.9 * state_value_arr[aaafter_a][aaafter_b]))
            else:
                aaafter_a = min(20, 3 + aafter_a)
                aaafter_b = min(20, 2 + aafter_b)
                # print(aaafter_a, aaafter_b)
                # 计算V(s')
                state_value += posibility_len * (profit + 0.9 * state_value_arr[aaafter_a][aaafter_b])

    return state_value


def drawTable(para_table, para_id, size):
    plot1, _ = plt.subplots()

    # 支持输入 np.array 传入策略table 数值 -5~5
    table = para_table

    # 设置X轴选项 ndarray
    x_tick = range(size)
    plt.xticks(np.arange(len(x_tick)), labels=x_tick, rotation=45, rotation_mode="anchor", ha="right")

    # 设置Y轴选项 ndarray
    y_tick = range(size)
    plt.yticks(np.arange(len(y_tick)), labels=y_tick)

    # 设置表名
    plt.title("This is a table ! ")

    # 设置块上数字 注意这个i j 与 table需要对齐
    for i in range(len(y_tick)):
        for j in range(len(x_tick)):
            plt.text(i, j, table[i, j], ha="center", va="center", color="w")

    # 设置配色
    plt.imshow(table, cmap=plt.jet())
    plt.colorbar()
    plt.tight_layout()
    # show 之前保存
    plt.savefig('./figures/action_iter_%d.png' % para_id)
    # plt.show()
    plt.close(plot1)


if __name__ == '__main__':
    print("begin\n")

    # 0. 初始状态 维护 维护状态值函数表  维护动作值函数表 11 是因为最多11个选择-5 - 5
    #  维护贪心策略表，记录当前最优策略 允许多个act同时为 1 允许软策略
    state_value_arr = np.zeros([21, 21], dtype=float)
    act_arr = np.zeros([21, 21], dtype=int)

    # 双重总循环 迭代进行策略评估与 策略改进 在连续两次策略相同时候结束
    iter_num = 0
    while True:
        print("iter: %d calculate each Value " % iter_num)

        # 策略评估 在收敛时退出 ———— 插值小于delta
        old_value = 0
        delta = 0.001
        iter_a = 0
        len_A_pro_list = getPoissonRate(3).tolist()
        len_B_pro_list = getPoissonRate(4).tolist()
        re_A_pro_list = getPoissonRate(3).tolist()
        re_B_pro_list = getPoissonRate(2).tolist()
        while True:
            sum_value = 0
            for numA in range(21):
                # print("Line: %d over " % numA)
                for numB in range(21):
                    new_value = getStateValue(numA, numB, state_value_arr, act_arr[numA][numB],
                                              len_A_pro_list, len_B_pro_list,
                                              re_A_pro_list, re_B_pro_list)
                    sum_value += abs(state_value_arr[numA][numB] - new_value)
                    state_value_arr[numA][numB] = new_value

            if abs(sum_value - old_value) <= delta:
                print(abs(sum_value - old_value))
                break
            else:
                iter_a = iter_a + 1
                if iter_a % 1 == 0:
                    print(abs(sum_value - old_value))
                old_value = sum_value

        # 进行策略改进
        action_value_arr = np.zeros([21, 21, 11], dtype=float)
        for numA in range(21):
            for numB in range(21):
                for act in range(-5, 6, 1):
                    act_index = act + 5
                    # 如果实际搬动数量小于操作数量，则说明该操作不可执行
                    lift_num, _, _ = getCost(numA, numB, act)
                    if lift_num < abs(act):
                        action_value_arr[numA][numB][act_index] = -np.inf
                    else:
                        action_value_arr[numA][numB][act_index] = getStateValue(numA, numB, state_value_arr, act,
                                                                                len_A_pro_list, len_B_pro_list,
                                                                                re_A_pro_list, re_B_pro_list)

        index_of_best = np.argmax(action_value_arr, axis=2)
        old_act_arr = act_arr
        act_arr = index_of_best - 5
        tables = np.argmax(action_value_arr, axis=2) - 5
        print(tables)
        drawTable(tables, iter_num, tables.shape[0])
        if (old_act_arr == act_arr).all() :
            print("iter: %d policy stable! " % iter_num)
            drawTable(tables, 99999, tables.shape[0])
            break

        iter_num += 1

    # 2. 进行策略评估 ———— 对每个 V 由策略 act_arr（策略） 进行迭代计算
    # 其中一次迭代： V(s1) -> sigema action（策略） -> rent+reward -> reuse+car;
    #               V(s2) -> sigema action（策略） -> rent+reward -> reuse+car;
    # 2. 计算动作值函数 对每个 V[i][j] 计算 [11]个动作下的新的 V 并赋值
