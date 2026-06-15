from datetime import datetime


def bookLeave(employee_name, start_date, end_date, reason):
    """
    Books leave for an employee based on the provided details.
    
    Args:
        employee_name (str): The name of the employee requesting leave.
        start_date (str): The start date of the leave in YYYY-MM-DD format.
        end_date (str): The end date of the leave in YYYY-MM-DD format.
        reason (str): The reason for the leave request.
    Returns:
        str: A confirmation message indicating the leave has been booked.
    """
    # Here you would typically have logic to save the leave request to a database
    # For this example, we'll just return a confirmation message
    return f"Leave booked for {employee_name} from {start_date} to {end_date} for reason: {reason}"