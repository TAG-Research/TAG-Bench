import heapq

import pandas as pd


class IndexMerger:
    class HeapElement:
        def __init__(self, distance, row):
            self.distance = distance
            self.row = row
            self.reranker_score = None

        def __lt__(self, other):
            return self.distance < other.distance

    def __init__(self, db_table_pairs, model):
        self.model = model
        self.db_table_pairs = db_table_pairs
        self.all_dfs = []
        for db, table in db_table_pairs:
            self.all_dfs.append(pd.read_csv(f"../pandas_dfs/{db}/{table}.csv"))

    def __call__(self, query, k):
        heap = []
        for i, (db, table) in enumerate(self.db_table_pairs):
            self.model.load_index(f"../indexes/{db}/{table}")
            df = self.all_dfs[i]
            distances, idxs = self.model(query, k)
            idxs = idxs[0]
            distances = distances[0]

            for idx, distance in zip(idxs, distances):
                if len(heap) < k:
                    heapq.heappush(heap, self.HeapElement(distance, df.iloc[idx]))
                else:
                    heapq.heappushpop(heap, self.HeapElement(distance, df.iloc[idx]))

        return heap


# Serialization format from STaRK - https://arxiv.org/pdf/2404.13207
def row_to_str(row):
    return "\n".join([f"- {col}: {val}" for col, val in row.items()])
