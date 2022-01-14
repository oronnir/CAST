from collections import defaultdict


class UnionFind:
    @staticmethod
    def _indices_dict(lis):
        d = defaultdict(list)
        for i, (a, b) in enumerate(lis):
            d[a].append(i)
            d[b].append(i)
        return d

    @staticmethod
    def _disjoint_indices(lis):
        d = UnionFind._indices_dict(lis)
        sets = []
        while len(d):
            que = set(d.popitem()[1])
            ind = set()
            while len(que):
                ind |= que
                que = set([y for i in que
                           for x in lis[i]
                           for y in d.pop(x, [])]) - ind
            sets += [ind]
        return sets

    @staticmethod
    def disjoint_sets(lis):
        return [set([x for i in s for x in lis[i]]) for s in UnionFind._disjoint_indices(lis)]
