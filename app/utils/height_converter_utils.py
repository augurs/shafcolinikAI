def convert_to_cm(feet: int = 0, inches: int = 0, centimeters: int = 0) -> int:
    
    if centimeters > 0:
        return centimeters

    if feet >= 0 and inches >= 0 and (feet > 0 or inches > 0):
        return round((feet * 30.48) + (inches * 2.54), 2)

    raise ValueError("Provide a non-zero height either in centimeters or feet/inches.")
