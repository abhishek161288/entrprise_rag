from langchain.tools import tool
from qdrant_client.models import date
from app.AIAgent.BookLeave.book_leave_service import (
    bookLeave
)

@tool("book_leave_tool")
def book_leave_tool(employee_name: str, start_date: str, end_date: str, reason: str) -> str:
    """
    Book leave for an employee based on the provided details.
    """
    print(f"Received leave booking request for {employee_name} from {start_date} to {end_date} for reason: {reason}")
    booking_details = bookLeave(employee_name, start_date, end_date, reason)
    print(f"Leave booking response: {booking_details}")
    return f"Leave booked successfully! Details: {booking_details}"