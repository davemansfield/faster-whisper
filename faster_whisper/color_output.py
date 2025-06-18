# Terminal color map. 10 colors grouped in ranges [0.0, 0.1, ..., 0.9]
# Lowest is red, middle is yellow, highest is green.
COLORS = [
    "\033[38;5;196m",  # Red
    "\033[38;5;202m",  # Orange-Red
    "\033[38;5;208m",  # Dark Orange
    "\033[38;5;214m",  # Orange
    "\033[38;5;220m",  # Gold
    "\033[38;5;226m",  # Yellow
    "\033[38;5;190m",  # Light Yellow
    "\033[38;5;154m",  # Light Green
    "\033[38;5;118m",  # Green
    "\033[38;5;82m",   # Bright Green
]

RESET = "\033[0m"

def get_color_for_probability(probability: float) -> str:
    """Get the appropriate color code for a given probability score.
    
    Probabilities are between 0 and 1 where:
    - 1.0 means 100% confidence (bright green)
    - 0.5 means 50% confidence (yellow)
    - 0.0 means 0% confidence (red)
    """
    # Ensure probability is between 0 and 1
    probability = max(0.0, min(1.0, probability))
    
    # Map probability to color index (0-9)
    color_index = int(probability * 9)
    return COLORS[color_index]

def colorize_text(text: str, probability: float) -> str:
    """Colorize text based on probability score."""
    color = get_color_for_probability(probability)
    return f"{color}{text}{RESET}"

def print_colored_segment(segment):
    """Print a segment with colored words based on their probabilities."""
    if not segment.words:
        print(segment.text)
        return

    colored_text = ""
    for word in segment.words:
        colored_text += colorize_text(word.word, word.probability)
        colored_text += " "  # Add space between words
    
    print(colored_text.strip())

def transcribe_with_colors(model, audio_path, **kwargs):
    """Wrapper function to transcribe audio with colored output based on confidence."""
    # Ensure word_timestamps is enabled
    kwargs['word_timestamps'] = True
    
    # Transcribe the audio
    segments, info = model.transcribe(audio_path, **kwargs)
    
    # Print colored output
    for segment in segments:
        print_colored_segment(segment)
    
    return segments, info 