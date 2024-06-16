class Statistics():
    '''
    统计记录类
    '''
    def __init__(self, is_record_dist=False) -> None:
        '''
            使用 Welford 方法更新均值与方差
            默认不记录分布，若需要记录分布则使用字典记录（等效于散列表）
        '''
        self.count = 0  # 数据数量
        self.mean = 0
        self.M2 = 0
        self.max = None
        self.min = None
        # 分布记录
        self.is_record_dist = is_record_dist
        if is_record_dist:
            self.dist = {}
    def update(self, x):
        '''记录新加入记录并更新当前统计量'''
        self.count += 1
        if self.max is None or x > self.max:
            self.max = x
        if self.min is None or x < self.min:
            self.min = x
        # 中间值更新
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2
        # 分布记录
        if self.is_record_dist:
            if x in self.dist:
                self.dist[x] += 1
            else:
                self.dist[x] = 1
    
    def __str__(self) -> str:
        return f"(mean={self.mean:.4f}, svar={self.svar:.4f})"
    def __repr__(self):
        return self.__str__()
    
    def __getattr__(self, key):
        if key == "var":
            if self.count < 2:
                return float('nan')
            return self.M2 / self.count
        elif key == "svar":
            if self.count < 2:
                return float('nan')
            return self.M2 / (self.count-1)
        elif key == "std":
            return (self.__getattr__("var"))**0.5
        else:
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {key!r}")
    def __add__(self, other):
        if not isinstance(other, Statistics):
            return NotImplemented
        result = Statistics(is_record_dist=self.is_record_dist and other.is_record_dist)
        if self.max is not None and other.max is not None:
            result.max = max(self.max, other.max)
            result.min = min(self.min, other.min)
        else:
            if self.max is not None:
                result.max = self.max
                result.min = self.min
            else:
                result.max = other.max
                result.min = other.min
        # 使用 Chan, Golub, and LeVeque 提出的方法进行合并
        result.count = self.count + other.count
        delta = other.mean - self.mean
        weighted_mean = self.mean + delta * other.count / result.count
        result.mean = weighted_mean
        result.M2 = self.M2 + other.M2 + delta**2 * self.count * other.count / result.count
        # 分布合并
        if result.is_record_dist:
            result.dist = self.dist.copy()
            for key, value in other.dist.items():
                if key in result.dist:
                    result.dist[key] += value
                else:
                    result.dist[key] = value
        return result

if __name__ == '__main__':
    import numpy as np

    # 测试单个统计对象
    data = np.random.rand(100)
    stats = Statistics()
    for number in data:
        stats.update(number)

    print("Statistics mean:", stats.mean)
    print("Numpy mean:", np.mean(data))
    print("Diff", stats.mean-np.mean(data))
    print("Statistics variance:", stats.svar)
    print("Numpy variance:", np.var(data, ddof=1))
    print("Diff", stats.svar-np.var(data, ddof=1))
    print("min max", stats.min, np.min(data), stats.max, np.max(data))

    # 测试两个统计对象的合并
    data1 = np.random.rand(100)
    data2 = np.random.rand(100)
    stats1 = Statistics()
    stats2 = Statistics()
    for number in data1:
        stats1.update(number)
    for number in data2:
        stats2.update(number)

    combined_stats = stats1 + stats2
    combined_data = np.concatenate((data1, data2))

    print("Combined Statistics mean:", combined_stats.mean)
    print("Combined Numpy mean:", np.mean(combined_data))
    print("Diff", combined_stats.mean-np.mean(combined_data))
    print("Combined Statistics variance:", combined_stats.svar)
    print("Combined Numpy variance:", np.var(combined_data, ddof=1))
    print("Diff", combined_stats.svar-np.var(combined_data, ddof=1))
    print("min max", combined_stats.min, np.min(combined_data), combined_stats.max, np.max(combined_data))