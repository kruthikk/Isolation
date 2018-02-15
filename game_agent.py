"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
from random import randint

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def valid_moves(game, player):
    move = game.get_player_location(player)
    r, c = move
    directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
    valid_moves = [(r + dr, c + dc) for dr, dc in directions if in_bounds(game, r + dr, c + dc)]
    return valid_moves


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    
    Outputs a score equal to the difference in the number of moves available to the
    two players with added bonus points if player is in center and bonus point if around the center
    
    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    credit = 0.0

    center = ((int(game.width/2), int(game.height/2)))
    r,c = center
    directions = [(-1,-2), (-1,2),(1, -2), (1, 2), (2, -1), (2, 1), (-2, -1), (-2, 1)]
    off_center = [(r + dr, c + dc) for dr, dc in directions if in_bounds(game, r + dr, c + dc)]
    player_location = game.get_player_location(player)
    if player_location == center:
        credit = 1.5
    elif player_location in off_center:
        credit = 0.5

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - opp_moves) + credit

    

def in_bounds(game, row, col):
    return 0 <= row  < game.height and 0 <= col < game.width

def winningmovescore(game, player):

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")    

    center = ((int(game.width/2), int(game.height/2)))
    player_location = game.get_player_location(player)
    opponent_location = game.get_player_location(game.get_opponent(player))
    if (player_location == center):
        return float("inf")
    else:
        corners={(0,0), (0, game.width-1), (game.height-1,0), (game.width-1, game.height-1)}
        if opponent_location in corners and player_location in corners:
            return float("inf")
        else:
            return float(len(game.get_legal_moves(player)) - len(game.get_legal_moves(game.get_opponent(player))))


def OutOfAreaScore(game,player):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))

    if (opp_moves==0):
        return float("inf")

    if all(item in opp_moves for item in own_moves):
        return float(len(own_moves) - len(opp_moves))
    else:
        if (own_moves == opp_moves):
            if (game._active_player == player):
                return float("-inf")
        if len(own_moves) > len(opp_moves):
            return float("inf")
        else:
             return float("-inf")

    return float(len(own_moves) - len(opp_moves))

def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    checks if the player occupies the center space or the player is in the diagonally opposite space of 
    opponent and if not returns the difference in the no of legal moves available
    
    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")    

    center = ((int(game.width/2), int(game.height/2)))
    player_location = game.get_player_location(player)
    opponent_location = game.get_player_location(game.get_opponent(player))
    if (player_location == center):
        return float("inf")
    else:
        corners={(0,0), (0, game.width-1), (game.height-1,0), (game.width-1, game.height-1)}
        if opponent_location in corners and player_location in corners:
            return float("inf")
        else:
            return float(len(game.get_legal_moves(player)) - len(game.get_legal_moves(game.get_opponent(player))))


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    
    Determines Manhattan distance from player to oppornent

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_location = game.get_player_location(player)
    opp_location = game.get_player_location(game.get_opponent(player))
    distance = abs(own_location[0]-opp_location[0]) + abs(own_location[1]-opp_location[1])
    return float(distance)




class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)
            #return self.minimax(game, 3)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!
       
        best_move = (-1,-1)
        moves = game.get_legal_moves()
        if  not moves :
            return best_move

        best_move = moves[0]
        best_score = float('-inf')

        for move in moves:
            mydepth= depth
            clone = game.forecast_move(move)
            score = self.min_value(clone, mydepth-1)
            if score > best_score:
                best_move = move
                best_score = score
        return best_move

    
    def min_value(self, game, depth):
    
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        moves = game.get_legal_moves()
        if not moves or depth <= 0:
            return self.score(game,self)
        
        best_score = float('inf')
        for move in moves:
            mydepth= depth
            clone = game.forecast_move(move)
            score = self.max_value(clone, mydepth-1)
            if score < best_score:
                best_move = move
                best_score = score
        return best_score

    def max_value(self, game, depth):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        moves = game.get_legal_moves()
        if not moves or depth <= 0:
            return self.score(game, self)
        
        best_score = float('-inf')
        for move in moves:
            mydepth= depth
            clone = game.forecast_move(move)
            score = self.min_value(clone, mydepth-1)
            if score > best_score:
                best_move = move
                best_score = score
        return best_score                   

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            
            #return self.alphabeta(game, self.search_depth)
            #return self.alphabeta(game, 3)
            
            moves = game.get_legal_moves()
            if not moves:
                return best_move

            if len(moves) == 1:
                return moves[0]

            depth = 1
            currentdepthmove = best_move
            while True:
                currentdepthmove =  self.alphabeta(game, depth)
                best_move = currentdepthmove
                depth += 1
        
        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed
        
        # Return the best move from the last completed search iteration
        return best_move

    
    def min_value(self, game, depth, alpha, beta):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        best_move=(-1,-1)
        moves = game.get_legal_moves()
        if not moves or depth == 0:
            return self.score(game, self)
        
        
        best_score = float('inf')
        for move in moves:
            movedepth= depth
            clone = game.forecast_move(move)
            best_score = min(best_score, self.max_value(clone, movedepth-1, alpha, beta))
            beta = min(beta, best_score)
            if best_score <= alpha:
                return best_score
            
        return best_score

    def max_value(self, game, depth, alpha, beta):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        best_move=(-1,-1)
        
        moves = game.get_legal_moves()
        if not moves or depth == 0:
            return self.score(game, self)
            
        best_score = float('-inf')
        for move in moves:
            movedepth= depth
            clone = game.forecast_move(move)
            best_score = max(best_score, self.min_value(clone, movedepth-1, alpha, beta))
            alpha = max(alpha, best_score)
            if best_score >= beta:
                return best_score
            
        return best_score


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):

        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!

        #action, state = np.argmax(game.get_legal_moves(), lambda a, s: max_value(s, depth, -infinity, infinity))
        #return action
        
        maxvalueslist= {}


        best_move = (-1,-1)
        moves = game.get_legal_moves()
        if  not moves :
            return best_move

        for move in moves:
            clone =  game.forecast_move(move)
            score = self.min_value(clone, depth-1, alpha, beta)
            maxvalueslist[move] = score
            alpha = max(score, alpha)
            

        best_move = max(maxvalueslist, key=lambda key: maxvalueslist[key])
        return best_move
        
     


    

