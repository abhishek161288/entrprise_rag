from datetime import datetime


def create_booking(
    office,
    date
):

    return {
        "booking_id":
            "CAB123",

        "office":
            office,

        "date":
            date,

        "status":
            "CONFIRMED",

        "pickup":
            "08:30 AM"
    }