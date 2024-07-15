def are_symmetric(state1: str, state2: str) -> bool:
    """
    state1 is the solution obtained for the optimization;
    state2 is the solution obtained by transferring the parameters.
    """
    flipped_state1 = "".join('1' if bit == '0' else '0' for bit in state1)
    return flipped_state1 == state2 
