class Match:
    def __init__(self, t1_rank, t1_form_wins, t1_form_losses, t1_h2h_wins, t2_rank, t2_form_wins,
                 t2_form_losses, t2_h2h_wins, h2h_ot, t1_score, t2_score, res):
        self.t1_rank = t1_rank
        self.t1_form_wins = t1_form_wins
        self.t1_form_losses = t1_form_losses
        self.t1_h2h_wins = t1_h2h_wins
        self.t2_rank = t2_rank
        self.t2_form_wins = t2_form_wins
        self.t2_form_losses = t2_form_losses
        self.t2_h2h_wins = t2_h2h_wins
        self.h2h_ot = h2h_ot
        self.t1_score = t1_score
        self.t2_score = t2_score
        self.res = res

    """
    Get team 1's form.
    Form is defined by their recent wins - recent losses
    """
    def get_t1_form(self):
        return self.t1_form_wins - self.t1_form_losses

    """
    Get team 2's form.
    Form is defined by their recent wins - recent losses
    """
    def get_t2_form(self):
        return self.t2_form_wins - self.t2_form_losses

    """
    Get the teams previous h2h performance.
    Positive is in favor of team 1, negative is in favor of team 2.
    This returns team 1's h2h wins - team 2's
    """
    def get_h2h(self):
        return self.t1_h2h_wins - self.t2_h2h_wins

    """
    Get the ranking of team 1, if they don't have a rank, this returns 300, a number close to the lowest possible
    ranking.
    """
    def get_t1_rank(self):
        if self.t1_rank == 0:
            return 300
        return self.t1_rank

    """
    Get the ranking of team 2, if they don't have a rank, this returns 300, a number close to the lowest possible
    ranking.
    """
    def get_t2_rank(self):
        if self.t2_rank == 0:
            return 300
        return self.t2_rank
