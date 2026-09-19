"""Fixed synthetic discovery, development, calibration and test inputs."""
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def build():
    discovery_items = ["meal", "concert", "vacation", "service", "film", "purchase", "class", "game"]
    contrasts = [("wonderful", "awful"), ("excellent", "terrible"),
                 ("delightful", "disappointing"), ("enjoyable", "unpleasant"),
                 ("great", "bad"), ("superb", "dreadful"), ("impressive", "poor"),
                 ("satisfying", "frustrating"), ("pleasant", "horrible"), ("amazing", "disgusting"),
                 ("brilliant", "mediocre"), ("outstanding", "disastrous")]
    discovery = [{"positive": f"My experience with the {item} was {pos}",
                  "negative": f"My experience with the {item} was {neg}"}
                 for item in discovery_items for pos, neg in contrasts]
    dev_items = ["hotel room", "headphones", "lunch", "museum tour", "novel", "camera",
                 "cooking lesson", "board game", "train ride", "coffee shop", "jacket", "theater show"]
    dev_templates = ["I tried the {item} yesterday. My overall impression is that it",
                     "Here is my review of the {item}. After trying it, I thought it",
                     "I spent some time with the {item}. In my opinion, it"]
    dev = [{"id": f"dev_review_{i:02d}", "kind": "review", "item": item,
            "prompt": dev_templates[i % 3].format(item=item)} for i, item in enumerate(dev_items)]
    test_items = ["wireless keyboard", "desk lamp", "backpack", "running shoes", "electric kettle",
        "vacuum cleaner", "bicycle helmet", "travel pillow", "raincoat", "garden chair", "wristwatch",
        "drawing tablet", "water bottle", "sleeping bag", "alarm clock", "notebook", "laptop stand",
        "portable speaker", "tennis racket", "yoga mat", "blender", "toaster", "table fan", "suitcase",
        "puzzle book", "chess set", "picnic basket", "fountain pen", "hiking boots", "swimming goggles",
        "office chair", "bedside table", "walking tour", "boat trip", "art exhibition", "pottery workshop",
        "dance lesson", "language course", "camping trip", "aquarium visit", "weekend retreat", "bus tour",
        "guided hike", "bakery visit", "breakfast buffet", "afternoon tea", "pizza delivery", "dinner cruise",
        "science exhibit", "photography course", "bowling session", "escape room", "book club", "jazz evening",
        "sports lesson", "festival visit", "rental apartment", "airport lounge", "haircut", "repair service",
        "fitness class", "music lesson", "ferry crossing", "countryside stay"]
    templates = ["Customer review of the {item}: I had a chance to try it and found it",
        "My thoughts on the {item}: Based on my experience, it was",
        "Personal review: the {item}. Having tried it myself, I would say it",
        "I want to describe my experience with the {item}. For me, it was",
        "An honest review of the {item}: The experience left me feeling",
        "Review notes about the {item}: Looking back, I think it was",
        "After spending time with the {item}, here is my opinion. I found it",
        "This is my account of trying the {item}. Overall, I felt it was"]
    test = [{"id": f"test_review_{i:02d}", "kind": "review", "item": item,
             "prompt": templates[i % 8].format(item=item)} for i, item in enumerate(test_items)]
    demos = ("Fact: The box is red.\nQuestion: What color is the box?\nAnswer: red\n\n"
             "Fact: Maya lives in Rome.\nQuestion: Where does Maya live?\nAnswer: Rome\n\n"
             "Fact: The pet is a dog.\nQuestion: What animal is the pet?\nAnswer: dog\n\n"
             "Fact: The owner is Leo.\nQuestion: Who is the owner?\nAnswer: Leo\n\n")
    colors = ["blue", "green", "yellow", "black", "white", "purple", "orange", "brown"]
    cities = ["Paris", "London", "Berlin", "Madrid", "Tokyo", "Dublin", "Vienna", "Oslo"]
    animals = ["cat", "rabbit", "horse", "bird", "fish", "mouse", "turtle", "hamster"]
    names = ["Anna", "David", "Sarah", "James", "Emma", "Daniel", "Laura", "Peter"]
    nouns = ["bag", "car", "hat", "cup", "shirt", "door", "chair", "ball"]

    def qa(n, split):
        result = []
        for i in range(n):
            k, j = i % 4, i // 4
            answer = [colors, cities, animals, names][k][(j + (3 if split == "dev" else 0)) % 8]
            obj = nouns[(j // 8 + j + (2 if split == "dev" else 0)) % 8]
            who = names[(j + 2) % 8]
            facts = [(f"The {obj} is {answer}.", f"What color is the {obj}?"),
                     (f"{who} lives in {answer}.", f"Where does {who} live?"),
                     (f"The animal in the {obj} picture is a {answer}.", f"What animal is in the {obj} picture?"),
                     (f"The {obj} belongs to {answer}.", f"Who does the {obj} belong to?")]
            fact, question = facts[k]
            if split == "test":
                fact = f"In record {j + 100}, " + fact[0].lower() + fact[1:]
            result.append({"id": f"{split}_qa_{i:02d}", "kind": "qa", "answer": answer,
                           "prompt": demos + f"Fact: {fact}\nQuestion: {question}\nAnswer:"})
        return result

    calibration = []
    positive = ["I loved every minute of it.", "It exceeded all my expectations.",
                "I am very happy with it and would recommend it.", "A fantastic experience that I would gladly repeat."]
    negative = ["I hated every minute of it.", "It failed to meet my expectations.",
                "I am very unhappy with it and cannot recommend it.", "A dreadful experience that I would never repeat."]
    for item in ["book", "restaurant", "product", "trip"]:
        for label, endings in [(1, positive), (0, negative)]:
            for end in endings:
                calibration.append({"text": f"My review of the {item}: {end}", "label": label})
    result = {"discovery": discovery, "development": dev + qa(32, "dev"),
              "test": test + qa(64, "test"), "calibration": calibration,
              "review_seeds": [20260918, 20260919], "development_seed": 20260917}
    assert len({x["prompt"] for x in result["development"] + result["test"]}) == 172
    return result


if __name__ == "__main__":
    p = Path(__file__).parent / "prompts.json"
    data = build()
    p.write_text(json.dumps(data, indent=2) + "\n")
    print(p, digest(data))
