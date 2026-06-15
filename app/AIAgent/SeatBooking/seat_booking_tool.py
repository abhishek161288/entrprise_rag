from langchain.tools import tool
from app.AIAgent.SeatBooking.seat_booking_service import (
    create_booking
)


@tool("seat_booking")
def seat_booking(office: str, date: str) -> str:
    """
    Book a seat for the given office and date.
    """
    booking_details = create_booking(office, date)
    return f"Seat booked successfully! Details: {booking_details}"