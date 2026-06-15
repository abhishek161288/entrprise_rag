from langchain.tools import tool
from app.AIAgent.CabBooking.cab_booking_service import (
    reserve_seat
)

@tool("cab_booking")
def cab_booking(office: str, date: str) -> str:
    """
    Book a cab for the given office and date.
    """
    booking_details = reserve_seat(office, date)
    return f"Cab booked successfully! Details: {booking_details}"