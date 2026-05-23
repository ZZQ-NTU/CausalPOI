import json


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_checkin_dict(data_dict):
    return {
        pid: {w: c for w, c in zip(weeks, checkins)}
        for pid, (checkins, weeks, _, _, _) in data_dict.items()
    }


def get_all_weeks(data_dict):
    return sorted({
        w
        for _, (checkins, weeks, _, _, _) in data_dict.items()
        for w in weeks
    })
