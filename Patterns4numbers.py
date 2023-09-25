#!/usr/bin/env python3

#  c (или же C) -- массив типов сдвига соседних точек: "Направо и вверх" или просто "направо".сложность O(n)
#
#  n -- число элементов в С, по идее их на 1 меньше, чем точек, что не сходится с формулой, это не очень, но решаемо. сложность O(n)
#
#  q -- минимальный период последовательности. В моей интерпретации имеет сложность O(n).
#  Проверял на разных последовательностях, эта часть рабочая.
#
#  p -- сумма с_i на одном периоде, сложность O(n)
#
#  s -- такое число,что выполняется формула для с_i. В силу теории оно должно быть единственным. Сложность - O(n^2). Чтобы найти s, нужно проверить q^2 равенств вида f(a,i)=c_i (i пробегает от 1 до q-1, a пробегает от 0 до q-1). Я пробовал придумать алгоритм,который будет решать экваивалентные (q - 1) систем из 2 неравенств, но т.к. каждая система имеет до q  решений, то сложность в худшем случае не меняется.
#
#  То есть всё кроме нахождения  s имеет сложность не более O(n). Почему при решении q уравнений получается напрямую не лучшая сложность я написал на бумажке, попробовав решить систему в общем виде. ссылка: https://www.dropbox.com/s/7hj642ctz38t55k/2023-08-05%2022.34.27.pdf?dl=0 . Это не доказывает, что нельзя добиться лучшей сложности, но методы базовой алгебры не помогают упростить алгоритм.
#
# Далее будет приведён алгоритм, который с помощью четвёрки (N, k, h, x0) находит s за линейное время. То есть четверка n, q, p, s может находиться быстро.
#
# Определения взяты из статьи Discrete Representation of Straight Lines

# In[ ]:


import math


def find_s(c: list[int], p: int, q: int) -> int:
    def calc_value(i: int, s: int) -> int:
        return math.floor((p / q) * (i - s)) - math.floor(
            (p / q) * (i - s - 1)
        )

    for s in range(q):
        valid = True
        for i in range(q):
            if c[i] != calc_value(i + 1, s):
                valid = False
                break
        if valid:
            return s
    return 0


#    return None


def find_c(points: list[tuple[int, int]]) -> list[int]:
    c: list[int] = []
    for i in range(1, len(points)):
        c.append(points[i][1] - points[i - 1][1])
    return c


def find_n(c: list[int]) -> int:
    return len(c)


def getZarr(arr: list[int], Z: list[int]) -> None:  # Z-функция
    n = len(arr)
    l, r, k = 0, 0, 0

    for i in range(1, n):
        if i > r:
            l = r = i
            while r < n and arr[r - l] == arr[r]:
                r += 1
            Z[i] = r - l
            r -= 1
        else:
            k = i - l
            if Z[k] < r - i + 1:
                Z[i] = Z[k]
            else:
                l = i
                while r < n and arr[r - l] == arr[r]:
                    r += 1
                Z[i] = r - l
                r -= 1


def find_q(arr: list[int]) -> int:
    l = len(arr)
    Z = [0 for _ in range(l)]
    getZarr(arr, Z)
    periods: list[int] = []

    for i in range(1, l):
        if Z[i] == (l - i):
            periods.append(i)

    periods.append(l)
    ans: int = min(periods)
    return ans


def find_p(c: list[int], q: int) -> int:
    return sum(c[:q])


def find_nqps(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    c = find_c(points)
    n = find_n(c)
    q = find_q(c)
    p = find_p(c, q)
    s = find_s(c, p, q)
    return n, q, p, s


# Задайте точки паттерна
points: list[tuple[int, int]] = [(0, 0), (1, 0), (2, 0), (3, 0)]

ans = find_nqps(points)
print(f"n, q, p, s =", ans)


#
#
# Если предметы выложены в ряд, то промежутков между ними на один меньше, чем предметов, поэтому n на 1 меньше, чем точек. Эта проблема решается за счет периодичности. Формулу y(x) я взял из статьи Discrete Representation of Straight Lines LEODORSTAND ARNOLDW.M.SMEULDERS

# In[ ]:


# видимо нужно ввести pip install matplotlib, но не знаю где
import math
import matplotlib.pyplot as plt


def calculate_y(x: int, q: int, p: int, s: int) -> int:
    return math.floor((x - s) * (p / q) + math.ceil(s * p / q))


def main(n: int, q: int, p: int, s: int) -> list[tuple[int, int]]:
    x_values = range(n + 1)
    y_values = [calculate_y(x, q, p, s) for x in x_values]

    plt.plot(x_values, y_values, marker="o")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"y(x) = [({p}/{q})(x - {s}) + [{s}({p}/{q})]]")
    plt.grid(True)
    plt.show()
    return [(x, y) for x, y in zip(x_values, y_values)]


# Задайте параметры
n = 8
q = 5
p = 3
s = 0

ans = main(n, q, p, s)
print(ans)


# Проверка для конкретного паттерна:

# In[ ]:


points: list[tuple[int, int]] = [
    (0, 0),
    (1, 0),
    (2, 1),
    (3, 1),
    (4, 2),
    (5, 3),
]
n, q, p, s = find_nqps(points)
print(n, q, p, s)
ans2 = main(n, q, p, s)
print(f"Вы сейчас написали следующие точки:", points)
print(f"Изначальными точками паттерна были:", ans2)
print(
    f"Eсли наверху написаны одинаковые массивы, то всё хорошо: цепочка DSLS --> Do-Sm --> DSLS корректна"
)


# Обоснование, что все DSS имеют либо 1 либо 2 тип паттернов: https://www.dropbox.com/scl/fi/h675z82kht7xawg1zmuxi/pattens4numbers-3.pdf?rlkey=h9wl91ppdy2xv0r8oh0jr7uvv&dl=0

# Поиск паттернов, что до r округление вниз, а далее ввверх:

# In[ ]:


import math
from typing import Dict


def find_seq1(
    k_1: int, k_2: int, b: int, n: int
) -> list[list[tuple[int, int]]]:
    points: Dict[int, list[tuple[int, int]]] = {}
    points2: list[int] = []
    for x in range(n + 1):
        y1 = math.ceil((k_1 * x / k_2) + b)
        y2 = math.floor((k_1 * x / k_2) + b + 1)
        if y2 - y1 == 1:
            points[x] = [(x, y1), (x, y2)]
            points2.append(x)
        else:
            points[x] = [(x, y1)]
    #    print('points', points)

    lines: list[list[tuple[int, int]]] = []
    for p in points2[::-1]:
        line: list[tuple[int, int]] = [
            v[0] if k <= p else v[-1] for k, v in points.items()
        ]
        lines.append(line)
    return lines


# Поиск паттернов, что до r округление вверх, а далее вниз:

# In[ ]:


import math


def find_seq2(
    k_1: int, k_2: int, b: int, n: int
) -> list[list[tuple[int, int]]]:
    points: Dict[int, list[tuple[int, int]]] = {}
    points2: list[int] = []
    for x in range(n + 1):
        y1 = math.ceil((k_1 * x / k_2) + b) - 1
        y2 = math.floor((k_1 * x / k_2) + b + 1) - 1
        if y2 - y1 == 1:
            points[x] = [(x, y1), (x, y2)]
            points2.append(x)
        else:
            points[x] = [(x, y1)]
    #    print('points', points)

    lines: list[list[tuple[int, int]]] = []
    for p in points2[::-1]:
        line: list[tuple[int, int]] = [
            v[-1] if k <= p else v[0] for k, v in points.items()
        ]
        lines.append(line)
    return lines


#  Далее проверка всех цепочек DSLS --> Do-Sm --> DSLS через НОК и напрямую:

# Через НОК:

# In[ ]:


# Вспоогательные функции для нахождения НОК чисел от 1 до N (написан быстрый способ для N<10 000 и медленный для N>10 000):


# Быстрый способ N<10 000
prime_numbers = [
    2,
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    53,
    59,
    61,
    67,
    71,
    73,
    79,
    83,
    89,
    97,
    101,
    103,
    107,
    109,
    113,
    127,
    131,
    137,
    139,
    149,
    151,
    157,
    163,
    167,
    173,
    179,
    181,
    191,
    193,
    197,
    199,
    211,
    223,
    227,
    229,
    233,
    239,
    241,
    251,
    257,
    263,
    269,
    271,
    277,
    281,
    283,
    293,
    307,
    311,
    313,
    317,
    331,
    337,
    347,
    349,
    353,
    359,
    367,
    373,
    379,
    383,
    389,
    397,
    401,
    409,
    419,
    421,
    431,
    433,
    439,
    443,
    449,
    457,
    461,
    463,
    467,
    479,
    487,
    491,
    499,
    503,
    509,
    521,
    523,
    541,
    547,
    557,
    563,
    569,
    571,
    577,
    587,
    593,
    599,
    601,
    607,
    613,
    617,
    619,
    631,
    641,
    643,
    647,
    653,
    659,
    661,
    673,
    677,
    683,
    691,
    701,
    709,
    719,
    727,
    733,
    739,
    743,
    751,
    757,
    761,
    769,
    773,
    787,
    797,
    809,
    811,
    821,
    823,
    827,
    829,
    839,
    853,
    857,
    859,
    863,
    877,
    881,
    883,
    887,
    907,
    911,
    919,
    929,
    937,
    941,
    947,
    953,
    967,
    971,
    977,
    983,
    991,
    997,
    1009,
    1013,
    1019,
    1021,
    1031,
    1033,
    1039,
    1049,
    1051,
    1061,
    1063,
    1069,
    1087,
    1091,
    1093,
    1097,
    1103,
    1109,
    1117,
    1123,
    1129,
    1151,
    1153,
    1163,
    1171,
    1181,
    1187,
    1193,
    1201,
    1213,
    1217,
    1223,
    1229,
    1231,
    1237,
    1249,
    1259,
    1277,
    1279,
    1283,
    1289,
    1291,
    1297,
    1301,
    1303,
    1307,
    1319,
    1321,
    1327,
    1361,
    1367,
    1373,
    1381,
    1399,
    1409,
    1423,
    1427,
    1429,
    1433,
    1439,
    1447,
    1451,
    1453,
    1459,
    1471,
    1481,
    1483,
    1487,
    1489,
    1493,
    1499,
    1511,
    1523,
    1531,
    1543,
    1549,
    1553,
    1559,
    1567,
    1571,
    1579,
    1583,
    1597,
    1601,
    1607,
    1609,
    1613,
    1619,
    1621,
    1627,
    1637,
    1657,
    1663,
    1667,
    1669,
    1693,
    1697,
    1699,
    1709,
    1721,
    1723,
    1733,
    1741,
    1747,
    1753,
    1759,
    1777,
    1783,
    1787,
    1789,
    1801,
    1811,
    1823,
    1831,
    1847,
    1861,
    1867,
    1871,
    1873,
    1877,
    1879,
    1889,
    1901,
    1907,
    1913,
    1931,
    1933,
    1949,
    1951,
    1973,
    1979,
    1987,
    1993,
    1997,
    1999,
    2003,
    2011,
    2017,
    2027,
    2029,
    2039,
    2053,
    2063,
    2069,
    2081,
    2083,
    2087,
    2089,
    2099,
    2111,
    2113,
    2129,
    2131,
    2137,
    2141,
    2143,
    2153,
    2161,
    2179,
    2203,
    2207,
    2213,
    2221,
    2237,
    2239,
    2243,
    2251,
    2267,
    2269,
    2273,
    2281,
    2287,
    2293,
    2297,
    2309,
    2311,
    2333,
    2339,
    2341,
    2347,
    2351,
    2357,
    2371,
    2377,
    2381,
    2383,
    2389,
    2393,
    2399,
    2411,
    2417,
    2423,
    2437,
    2441,
    2447,
    2459,
    2467,
    2473,
    2477,
    2503,
    2521,
    2531,
    2539,
    2543,
    2549,
    2551,
    2557,
    2579,
    2591,
    2593,
    2609,
    2617,
    2621,
    2633,
    2647,
    2657,
    2659,
    2663,
    2671,
    2677,
    2683,
    2687,
    2689,
    2693,
    2699,
    2707,
    2711,
    2713,
    2719,
    2729,
    2731,
    2741,
    2749,
    2753,
    2767,
    2777,
    2789,
    2791,
    2797,
    2801,
    2803,
    2819,
    2833,
    2837,
    2843,
    2851,
    2857,
    2861,
    2879,
    2887,
    2897,
    2903,
    2909,
    2917,
    2927,
    2939,
    2953,
    2957,
    2963,
    2969,
    2971,
    2999,
    3001,
    3011,
    3019,
    3023,
    3037,
    3041,
    3049,
    3061,
    3067,
    3079,
    3083,
    3089,
    3109,
    3119,
    3121,
    3137,
    3163,
    3167,
    3169,
    3181,
    3187,
    3191,
    3203,
    3209,
    3217,
    3221,
    3229,
    3251,
    3253,
    3257,
    3259,
    3271,
    3299,
    3301,
    3307,
    3313,
    3319,
    3323,
    3329,
    3331,
    3343,
    3347,
    3359,
    3361,
    3371,
    3373,
    3389,
    3391,
    3407,
    3413,
    3433,
    3449,
    3457,
    3461,
    3463,
    3467,
    3469,
    3491,
    3499,
    3511,
    3517,
    3527,
    3529,
    3533,
    3539,
    3541,
    3547,
    3557,
    3559,
    3571,
    3581,
    3583,
    3593,
    3607,
    3613,
    3617,
    3623,
    3631,
    3637,
    3643,
    3659,
    3671,
    3673,
    3677,
    3691,
    3697,
    3701,
    3709,
    3719,
    3727,
    3733,
    3739,
    3761,
    3767,
    3769,
    3779,
    3793,
    3797,
    3803,
    3821,
    3823,
    3833,
    3847,
    3851,
    3853,
    3863,
    3877,
    3881,
    3889,
    3907,
    3911,
    3917,
    3919,
    3923,
    3929,
    3931,
    3943,
    3947,
    3967,
    3989,
    4001,
    4003,
    4007,
    4013,
    4019,
    4021,
    4027,
    4049,
    4051,
    4057,
    4073,
    4079,
    4091,
    4093,
    4099,
    4111,
    4127,
    4129,
    4133,
    4139,
    4153,
    4157,
    4159,
    4177,
    4201,
    4211,
    4217,
    4219,
    4229,
    4231,
    4241,
    4243,
    4253,
    4259,
    4261,
    4271,
    4273,
    4283,
    4289,
    4297,
    4327,
    4337,
    4339,
    4349,
    4357,
    4363,
    4373,
    4391,
    4397,
    4409,
    4421,
    4423,
    4441,
    4447,
    4451,
    4457,
    4463,
    4481,
    4483,
    4493,
    4507,
    4513,
    4517,
    4519,
    4523,
    4547,
    4549,
    4561,
    4567,
    4583,
    4591,
    4597,
    4603,
    4621,
    4637,
    4639,
    4643,
    4649,
    4651,
    4657,
    4663,
    4673,
    4679,
    4691,
    4703,
    4721,
    4723,
    4729,
    4733,
    4751,
    4759,
    4783,
    4787,
    4789,
    4793,
    4799,
    4801,
    4813,
    4817,
    4831,
    4861,
    4871,
    4877,
    4889,
    4903,
    4909,
    4919,
    4931,
    4933,
    4937,
    4943,
    4951,
    4957,
    4967,
    4969,
    4973,
    4987,
    4993,
    4999,
    5003,
    5009,
    5011,
    5021,
    5023,
    5039,
    5051,
    5059,
    5077,
    5081,
    5087,
    5099,
    5101,
    5107,
    5113,
    5119,
    5147,
    5153,
    5167,
    5171,
    5179,
    5189,
    5197,
    5209,
    5227,
    5231,
    5233,
    5237,
    5261,
    5273,
    5279,
    5281,
    5297,
    5303,
    5309,
    5323,
    5333,
    5347,
    5351,
    5381,
    5387,
    5393,
    5399,
    5407,
    5413,
    5417,
    5419,
    5431,
    5437,
    5441,
    5443,
    5449,
    5471,
    5477,
    5479,
    5483,
    5501,
    5503,
    5507,
    5519,
    5521,
    5527,
    5531,
    5557,
    5563,
    5569,
    5573,
    5581,
    5591,
    5623,
    5639,
    5641,
    5647,
    5651,
    5653,
    5657,
    5659,
    5669,
    5683,
    5689,
    5693,
    5701,
    5711,
    5717,
    5737,
    5741,
    5743,
    5749,
    5779,
    5783,
    5791,
    5801,
    5807,
    5813,
    5821,
    5827,
    5839,
    5843,
    5849,
    5851,
    5857,
    5861,
    5867,
    5869,
    5879,
    5881,
    5897,
    5903,
    5923,
    5927,
    5939,
    5953,
    5981,
    5987,
    6007,
    6011,
    6029,
    6037,
    6043,
    6047,
    6053,
    6067,
    6073,
    6079,
    6089,
    6091,
    6101,
    6113,
    6121,
    6131,
    6133,
    6143,
    6151,
    6163,
    6173,
    6197,
    6199,
    6203,
    6211,
    6217,
    6221,
    6229,
    6247,
    6257,
    6263,
    6269,
    6271,
    6277,
    6287,
    6299,
    6301,
    6311,
    6317,
    6323,
    6329,
    6337,
    6343,
    6353,
    6359,
    6361,
    6367,
    6373,
    6379,
    6389,
    6397,
    6421,
    6427,
    6449,
    6451,
    6469,
    6473,
    6481,
    6491,
    6521,
    6529,
    6547,
    6551,
    6553,
    6563,
    6569,
    6571,
    6577,
    6581,
    6599,
    6607,
    6619,
    6637,
    6653,
    6659,
    6661,
    6673,
    6679,
    6689,
    6691,
    6701,
    6703,
    6709,
    6719,
    6733,
    6737,
    6761,
    6763,
    6779,
    6781,
    6791,
    6793,
    6803,
    6823,
    6827,
    6829,
    6833,
    6841,
    6857,
    6863,
    6869,
    6871,
    6883,
    6899,
    6907,
    6911,
    6917,
    6947,
    6949,
    6959,
    6961,
    6967,
    6971,
    6977,
    6983,
    6991,
    6997,
    7001,
    7013,
    7019,
    7027,
    7039,
    7043,
    7057,
    7069,
    7079,
    7103,
    7109,
    7121,
    7127,
    7129,
    7151,
    7159,
    7177,
    7187,
    7193,
    7207,
    7211,
    7213,
    7219,
    7229,
    7237,
    7243,
    7247,
    7253,
    7283,
    7297,
    7307,
    7309,
    7321,
    7331,
    7333,
    7349,
    7351,
    7369,
    7393,
    7411,
    7417,
    7433,
    7451,
    7457,
    7459,
    7477,
    7481,
    7487,
    7489,
    7499,
    7507,
    7517,
    7523,
    7529,
    7537,
    7541,
    7547,
    7549,
    7559,
    7561,
    7573,
    7577,
    7583,
    7589,
    7591,
    7603,
    7607,
    7621,
    7639,
    7643,
    7649,
    7669,
    7673,
    7681,
    7687,
    7691,
    7699,
    7703,
    7717,
    7723,
    7727,
    7741,
    7753,
    7757,
    7759,
    7789,
    7793,
    7817,
    7823,
    7829,
    7841,
    7853,
    7867,
    7873,
    7877,
    7879,
    7883,
    7901,
    7907,
    7919,
    7927,
    7933,
    7937,
    7949,
    7951,
    7963,
    7993,
    8009,
    8011,
    8017,
    8039,
    8053,
    8059,
    8069,
    8081,
    8087,
    8089,
    8093,
    8101,
    8111,
    8117,
    8123,
    8147,
    8161,
    8167,
    8171,
    8179,
    8191,
    8209,
    8219,
    8221,
    8231,
    8233,
    8237,
    8243,
    8263,
    8269,
    8273,
    8287,
    8291,
    8293,
    8297,
    8311,
    8317,
    8329,
    8353,
    8363,
    8369,
    8377,
    8387,
    8389,
    8419,
    8423,
    8429,
    8431,
    8443,
    8447,
    8461,
    8467,
    8501,
    8513,
    8521,
    8527,
    8537,
    8539,
    8543,
    8563,
    8573,
    8581,
    8597,
    8599,
    8609,
    8623,
    8627,
    8629,
    8641,
    8647,
    8663,
    8669,
    8677,
    8681,
    8689,
    8693,
    8699,
    8707,
    8713,
    8719,
    8731,
    8737,
    8741,
    8747,
    8753,
    8761,
    8779,
    8783,
    8803,
    8807,
    8819,
    8821,
    8831,
    8837,
    8839,
    8849,
    8861,
    8863,
    8867,
    8887,
    8893,
    8923,
    8929,
    8933,
    8941,
    8951,
    8963,
    8969,
    8971,
    8999,
    9001,
    9007,
    9011,
    9013,
    9029,
    9041,
    9043,
    9049,
    9059,
    9067,
    9091,
    9103,
    9109,
    9127,
    9133,
    9137,
    9151,
    9157,
    9161,
    9173,
    9181,
    9187,
    9199,
    9203,
    9209,
    9221,
    9227,
    9239,
    9241,
    9257,
    9277,
    9281,
    9283,
    9293,
    9311,
    9319,
    9323,
    9337,
    9341,
    9343,
    9349,
    9371,
    9377,
    9391,
    9397,
    9403,
    9413,
    9419,
    9421,
    9431,
    9433,
    9437,
    9439,
    9461,
    9463,
    9467,
    9473,
    9479,
    9491,
    9497,
    9511,
    9521,
    9533,
    9539,
    9547,
    9551,
    9587,
    9601,
    9613,
    9619,
    9623,
    9629,
    9631,
    9643,
    9649,
    9661,
    9677,
    9679,
    9689,
    9697,
    9719,
    9721,
    9733,
    9739,
    9743,
    9749,
    9767,
    9769,
    9781,
    9787,
    9791,
    9803,
    9811,
    9817,
    9829,
    9833,
    9839,
    9851,
    9857,
    9859,
    9871,
    9883,
    9887,
    9901,
    9907,
    9923,
    9929,
    9931,
    9941,
    9949,
    9967,
    9973,
]


def lcm_of_first_n_numbers1000(n: int, prime_numbers: list[int]) -> int:
    result = 1

    for i in range(n):
        prime = prime_numbers[i]
        k = int(math.log(n) / math.log(prime))
        result *= prime**k

    return result


n = int(input("Введите n: "))
lcm = lcm_of_first_n_numbers1000(n, prime_numbers)
print(f"НОК первых {n} натуральных чисел для N<10 000: {lcm}")


# Медленный способ N>10 000


def prime_factors_sieve(
    n: int,
) -> list[int]:  # Эта функция возвращает список простых чисел до n
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    primes: list[int] = []

    for p in range(2, int(math.sqrt(n)) + 1):
        if is_prime[p]:
            for i in range(p * p, n + 1, p):
                is_prime[i] = False

    for p in range(2, n + 1):
        if is_prime[p]:
            primes.append(p)

    return primes


def lcm_of_first_n_numbers_for_big(n: int):
    primes: list[int] = prime_factors_sieve(n)
    result = 1

    for prime in primes:
        k = int(math.log(n) / math.log(prime))
        result *= prime**k

    return result


n = int(input("Введите n: "))
lcm = lcm_of_first_n_numbers_for_big(n)
print(f"НОК первых {n} натуральных чисел для любого N: {lcm}")


#  Проверка всех цепочек DSLS --> Do-Sm --> DSLS для паттернов первого типа:

# In[ ]:


def check_for_n1(n: int, prime_numbers: list[int]) -> None:
    k2 = lcm_of_first_n_numbers1000(n, prime_numbers)
    for k1 in range(k2):
        lines = find_seq1(k1 + 1, k2, 0, n)
        for line in lines:
            n, q, p, s = find_nqps(line)
            print(n, q, p, s)
            ans2 = main(n, q, p, s)
            if ans2 != line:
                print("Error", line)


# In[ ]:


check_for_n1(3, prime_numbers)


# Проверка всех цепочек DSLS --> Do-Sm --> DSLS для паттернов второго типа:

# In[ ]:


def check_for_n2(n: int, prime_numbers: list[int]) -> None:
    k2: int = lcm_of_first_n_numbers1000(n, prime_numbers)
    for k1 in range(k2):
        lines = find_seq2(k1 + 1, k2, 0, n)
        for line in lines:
            n, q, p, s = find_nqps(line)
            print(n, q, p, s)
            ans2 = main(n, q, p, s)
            if ans2 != line:
                print("Error", line)


# In[ ]:


check_for_n2(3, prime_numbers)


# In[170]:


# Реализация N k h x0 из статьи A New Parametrization Of Digital Straight Line


def find_Ss(
    points: list[tuple[int, int]]
) -> tuple[
    list[tuple[int, int]], list[tuple[int, int]]
]:  # Находит верхнее и нижнее множество
    modified_points = points.copy()
    modified_points.append((len(points) - 1, 0))

    new_y_values = [point[1] + 1 for point in points]
    new_y_values.append(len(points))

    modified_points_with_new_y = list(
        zip([point[0] for point in points], new_y_values)
    )
    modified_points_with_new_y.insert(0, (0, len(points) - 1))

    return modified_points, modified_points_with_new_y


def T(
    a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]
) -> int:  # Находит sgn((b − a)⊥ · (c − b))
    cross_product = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (
        c[0] - b[0]
    )
    if cross_product > 0:
        return 1
    elif cross_product < 0:
        return -1
    else:
        return 0


def SeparatingCommonTangent(
    P0: list[tuple[int, int]], P1: list[tuple[int, int]]
) -> tuple[int, int] | None:  # Находит касательную
    n0, n1 = len(P0), len(P1)
    s0, t0, s1, t1, u_ind = 0, 1, 0, 1, 0

    while t0 <= 2 * n0 and t1 <= 2 * n1:
        if u_ind == 0:
            if T(P1[s1], P0[s0], P0[t0 % n0]) == 1:
                s0 = t0
                t1 = s1 + 1
            t0 = t0 + 1
            u_ind = 1 - u_ind
        else:
            if T(P0[s0], P1[s1], P1[t1 % n1]) == 1:
                s1 = t1
                t0 = s0 + 1
            t1 = t1 + 1
            u_ind = 1 - u_ind

    #    for t in range(n0):
    #        if T(P1[s1], P0[s0], P0[t]) == 1:
    #            return None
    #    for t in range(n1):
    #        if T(P0[s0], P1[s1], P1[t]) == 1:
    #            return None
    return s0, s1


def GetTangentPoints(P0: list[tuple[int, int]], P1: list[tuple[int, int]]):
    tangent_indices: tuple[int, int] | None = SeparatingCommonTangent(
        P0, P1
    )
    if tangent_indices is not None:
        s0, s1 = tangent_indices
        tangent_point_1 = P0[s0]
        tangent_point_2 = P1[s1]
        return tangent_point_1, tangent_point_2
    else:
        return (0, 0), (0, 0)


#    else:
#        return None


def find_x0(P_0: tuple[int, int]) -> int:
    return P_0[0]


def find_k(P_0: tuple[int, int], P_1: tuple[int, int]) -> int:
    return P_1[0] - P_0[0]


def find_h(P_0: tuple[int, int], P_1: tuple[int, int]) -> int:
    return P_1[1] - P_0[1]


def find_N(points: list[tuple[int, int]]) -> int:
    return len(points)


def find_Nkhx0(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    N = find_N(points)
    S0, S1 = find_Ss(points)
    result = GetTangentPoints(S0, S1)
    if result is not (None, None):
        p0, p1 = result
    else:
        p0 = (0, 0)
        p1 = (0, 0)
    x0 = find_x0(p0)
    k = find_k(p0, p1)
    h = find_h(p0, p1)
    return N, k, h, x0


# In[ ]:


# Визуализация N k h x0
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

# Задаем параметры прямой y = kx + n
k = 1/3
n = 0

# Генерируем целочисленные точки на прямой с округлением y
x_vals = np.arange(0, 11)
y_vals = np.round(k * x_vals + n)
points = list(zip(x_vals, y_vals))
print(f'points', points)

# Создаем график
plt.figure(figsize=(10, 6))
plt.scatter(*zip(*points), color='blue', label='Исходные точки')

# Визуализируем S0 и S1 как многоугольники
S0, S1 = find_Ss(points)
print(f'S0:',S0)
print(f'S1:',S1)
polygon_S0 = Polygon(S0, edgecolor='green', facecolor='lightgreen', label='S0')
polygon_S1 = Polygon(S1, edgecolor='orange', facecolor='lightcoral', label='S1')
plt.gca().add_patch(polygon_S0)
plt.gca().add_patch(polygon_S1)

# Находим общую касательную
P0, P1 = GetTangentPoints(S0, S1)
if P0 is not None and P1 is not None:
    plt.plot(*zip(P0, P1), linestyle='--', color='red', label='Общая касательная')

# Подписываем оси и т д
plt.xlabel('x')
plt.ylabel('y')
plt.title('Визуализация опорных точек S0 и S1, и общей касательной')
plt.legend()

# Рисуем график
plt.grid(True)
plt.show()
"""


# Проверка, что (n,q,p,s) и (n,q,p,x0) задают одни и те же паттерны:

# In[171]:


def check_for_n_x0ands(n: int, prime_numbers: list[int]) -> None:
    y = 0
    k2 = lcm_of_first_n_numbers1000(n, prime_numbers)
    for k1 in range(k2):
        lines = find_seq1(k1 + 1, k2, 0, n)
        for line in lines:
            n, q, p, s = find_nqps(line)
            print(n, q, p, s)
            _, _, _, x0 = find_Nkhx0(line)
            print(n, q, p, x0)
            ans2 = main(n, q, p, s)
            ans3 = main(n, q, p, x0)
            if ans2 != ans3:
                print("Error", line)
                y = 1

    for k1 in range(k2):
        lines = find_seq2(k1 + 1, k2, 0, n)
        for line in lines:
            n, q, p, s = find_nqps(line)
            print(n, q, p, s)
            _, _, _, x0 = find_Nkhx0(line)
            print(n, q, p, x0)
            ans2 = main(n, q, p, s)
            ans3 = main(n, q, p, x0)
            if ans2 != ans3:
                print("Error", line)
                y = 1
    if y == 1:
        print(";(")
    else:
        print(";)")


# In[172]:


check_for_n_x0ands(3, prime_numbers)


# Проверка, что x0 % q = s:

# In[173]:


def check_s_equal_x0modq(n: int):
    y = 0
    k2 = lcm_of_first_n_numbers1000(n, prime_numbers)
    for k1 in range(k2):
        lines = find_seq1(k1 + 1, k2, 0, n)
        for line in lines:
            n, q, _, s = find_nqps(line)
            #            print(n,q,p,s)
            _, _, _, x0 = find_Nkhx0(line)
            print(s % q, x0 % q)
            if (s % q) != (x0 % q):
                print("Error", line)
                y = 1
    for k1 in range(k2):
        lines = find_seq2(k1 + 1, k2, 0, n)
        for line in lines:
            n, q, _, s = find_nqps(line)
            #            print(n,q,p,s)
            _, _, _, x0 = find_Nkhx0(line)
            print(s % q, x0 % q)
            if (s % q) != (x0 % q):
                print("Error", line)
                y = 1

    if y == 1:
        print(";(")
    else:
        print(";)")


# In[174]:


check_s_equal_x0modq(3)


# Напрямую:

# Перебор всех дробей k1/k2<=1, где k2<=n:

# In[175]:


from fractions import Fraction


def generate_fractions(n: int) -> list[Fraction]:
    fractions_set: set[Fraction] = set()

    for k2 in range(1, n + 1):
        for k1 in range(k2 + 1):
            fraction: Fraction = Fraction(k1, k2)
            fractions_set.add(fraction)

    fractions_list = list(fractions_set)
    fractions_list.sort()

    return fractions_list


n = int(input("Введите значение n: "))
fractions = generate_fractions(n)

for fraction in fractions:
    k1, k2 = fraction.numerator, fraction.denominator
    print(f"{k1}, {k2}")


# Проверка всех цепочек DSLS --> Do-Sm --> DSLS для паттернов первого типа:

# In[176]:


def new_check_for_n1(n: int):
    for fraction in fractions:
        k1, k2 = fraction.numerator, fraction.denominator
        lines = find_seq1(k1 + 1, k2, 0, n)
        for line in lines:
            n, q, p, s = find_nqps(line)
            print(n, q, p, s)
            ans2 = main(n, q, p, s)
            if ans2 != line:
                print("Error", line)


# In[177]:


new_check_for_n1(3)


# Проверка всех цепочек DSLS --> Do-Sm --> DSLS для паттернов второго типа:

# In[178]:


def new_check_for_n2(n: int):
    for fraction in fractions:
        k1, k2 = fraction.numerator, fraction.denominator
        lines: list[list[tuple[int, int]]] = find_seq2(k1 + 1, k2, 0, n)
        for line in lines:
            n, q, p, s = find_nqps(line)
            print(n, q, p, s)
            ans2 = main(n, q, p, s)
            if ans2 != line:
                print("Error", line)


# In[158]:


new_check_for_n2(3)
