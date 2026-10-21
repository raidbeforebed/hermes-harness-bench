from app.pricing import total


def summarise(orders, rate):
    return [{"id": o["id"], "total": total(o["items"], rate)} for o in orders]
