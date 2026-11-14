import json
import math
import random

random.seed(42)


class XORNetwork:

    def __init__(self):
        self.lr_speed = 0.2
        self.w11 = random.uniform(-1, 1)
        self.w12 = random.uniform(-1, 1)
        self.b1 = random.uniform(-1, 1)

        self.w21 = random.uniform(-1, 1)
        self.w22 = random.uniform(-1, 1)
        self.b2 = random.uniform(-1, 1)

        self.w_out_1 = random.uniform(-1, 1)
        self.w_out_2 = random.uniform(-1, 1)
        self.b_out = random.uniform(-1, 1)

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, x1, x2):
        self.x1 = x1
        self.x2 = x2
        self.h1_input = (x1 * self.w11) + (x2 * self.w12) + self.b1
        self.h1_out = self.sigmoid(self.h1_input)

        self.h2_input = (x1 * self.w21) + (x2 * self.w22) + self.b2
        self.h2_out = self.sigmoid(self.h2_input)

        self.out_input = (
            (self.h1_out * self.w_out_1) + (self.h2_out * self.w_out_2) + self.b_out
        )
        self.final_out = self.sigmoid(self.out_input)
        return self.final_out

    def train(self, target):
        out_error = target - self.final_out
        out_delta = out_error * self.sigmoid_derivative(self.final_out)

        h1_error = out_delta * self.w_out_1
        h1_delta = h1_error * self.sigmoid_derivative(self.h1_out)

        h2_error = out_delta * self.w_out_2
        h2_delta = h2_error * self.sigmoid_derivative(self.h2_out)

        self.w_out_1 += self.lr_speed * out_delta * self.h1_out
        self.w_out_2 += self.lr_speed * out_delta * self.h2_out
        self.b_out += self.lr_speed * out_delta

        self.w11 += self.lr_speed * h1_delta * self.x1
        self.w12 += self.lr_speed * h1_delta * self.x2
        self.b1 += self.lr_speed * h1_delta

        self.w21 += self.lr_speed * h2_delta * self.x1
        self.w22 += self.lr_speed * h2_delta * self.x2
        self.b2 += self.lr_speed * h2_delta



with open("orders_1000.json", "r", encoding="utf-8") as f:
    orders = json.load(f)


pair_counts = {}
item_counts = {}

for order in orders:
    ids = [str(item["product_id"]) for item in order["items"]]
    for id1 in ids:
        item_counts[id1] = item_counts.get(id1, 0) + 1
        for id2 in ids:
            if id1 != id2:
                pair = (id1, id2)
                pair_counts[pair] = pair_counts.get(pair, 0) + 1


train_data = []
for (id1, id2), count in pair_counts.items():
    x1 = min(item_counts[id2] / 100.0, 1.0)
    x2 = min(count / 15.0, 1.0)
    target = 1 if count >= 3 else 0
    train_data.append((x1, x2, target))

network = XORNetwork()
for epoc in range(1000):
    for x1, x2, target in train_data:
        network.forward(x1, x2)
        network.train(target)


def get_recommendations(cart, menu, top_n=3):
    cart_ids = [str(item["product_id"]) for item in cart]
    scored_items = []

    for dish in menu:
        dish_id = str(dish["product_id"])
        if dish_id in cart_ids:
            continue


        best_score = 0.0
        for c_id in cart_ids:
            x1 = min(item_counts.get(dish_id, 0) / 100.0, 1.0)
            x2 = min(pair_counts.get((c_id, dish_id), 0) / 15.0, 1.0)
            score = network.forward(x1, x2)
            if score > best_score:
                best_score = score

        scored_items.append((best_score, dish))

    scored_items.sort(key=lambda x: x[0], reverse=True)
    return [dish for score, dish in scored_items[:top_n]]
