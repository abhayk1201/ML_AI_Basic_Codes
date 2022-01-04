import random
import copy
import time

class Teeko2Player:
    """ An object representation for an AI game player for the game Teeko2.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a Teeko2Player object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    
    def is_drop_phase(state):
        """ To check if drop phase or continued game play.
        """
        count = 0
        for row in state:
            for item in row:
                if item != ' ':
                    count += 1
        if count == 8:
            return False
        return True 
    
    
    def succ(self, state):
        """
        Takes in a board state and returns a list of the legal successors
        """
        
        #To get locations of my_piece for current state.
        current_filled = []
        for i in range(len(state)):
            for j in range(len(state[0])):
                if state[i][j] == self.my_piece:
                    current_filled.append((i, j)) 
        
        #list of legal successors           
        succ_list = [] 
        
        #Continued game play after drop_phase is over.
        if len(current_filled) == 4:
            for (i,j) in current_filled:
                # move in all possible adjacent locations
                adj_locations = [[i, j + 1], [i, j - 1], [i + 1, j], [i - 1, j], [i + 1, j + 1], [i + 1, j - 1], [i - 1, j + 1],
                         [i - 1, j - 1]]
                # iterate over all possible directions and check if it is still within the board
                for next_location in adj_locations:
                    if 0 <= next_location[0] and next_location[0] < 5 and 0 <= next_location[1] and next_location[1] < 5 and state[next_location[0]][next_location[1]] == ' ':
                        #create state deepcopy and change the identified location as my_piece and empty old location
                        next_state = copy.deepcopy(state)
                        next_state[next_location[0]][next_location[1]] = self.my_piece
                        next_state[i][j] = ' ' 
                        #succ_list.append(((i,j), next_location, next_state))
                        succ_list.append(next_state)
                        
        # Drop Phase: adding a new piece of the current player's type to the board
        else:
            for i in range(5):
                for j in range(5):
                    if state[i][j] == ' ': 
                        next_state = copy.deepcopy(state)
                        next_state[i][j] = self.my_piece
                        #succ_list.append((None, (i,j),  next_state))
                        succ_list.append(next_state)
            # First turn in drop phase: Select Random location till you get an empty location
#             if len(current_filled) == 0:
#                 (row, col) = (random.randint(0, 4), random.randint(0, 4)) 
#                 while state[row][col] != ' ':
#                     (row, col) = (random.randint(0, 4), random.randint(0, 4))
#                 #create state deepcopy and change the identified location as my_piece
#                 next_state = copy.deepcopy(state)
#                 next_state[row][col] = self.my_piece  
#                 #succ_list.append((None, (row, col), next_state))
#                 succ_list.append(next_state)
            
#             else:
#                 for (i,j) in current_filled:
#                     # move in all possible adjacent locations or other locations which can 
#                     adj_locations = [[i, j + 1], [i, j - 1], [i + 1, j], [i - 1, j], [i + 1, j + 1], [i + 1, j - 1],
#                              [i - 1, j + 1], [i - 1, j - 1]]
#                     for next_location in adj_locations:
#                         if 0 <= next_location[0] and next_location[0] < 5 and 0 <= next_location[1] and next_location[1] < 5 and state[next_location[0]][next_location[1]] == ' ':
#                             next_state = copy.deepcopy(state)
#                             next_state[next_location[0]][next_location[1]] = self.my_piece
#                             #succ_list.append(((i,j), next_location, next_state))
#                             succ_list.append(next_state)
                
        return succ_list


    def check_pattern(self, state, location, which_piece):
            
        #Horizontal Pattern towards left
        i,j = location
        horz_patt_count = 0
        while j > 0:
            j = j - 1
            if state[i][j] == which_piece:
                horz_patt_count += 1
            elif state[i][j] != ' ':
                break       
        #Horizontal Pattern towards right
        i,j = location
        while j < 4:
            j = j + 1
            if state[i][j] == which_piece:
                horz_patt_count += 1
            elif state[i][j] != ' ':
                break

        
        #Vertical Pattern towards top
        i,j = location
        vert_patt_count = 0
        while i > 0:
            i = i - 1
            if state[i][j] == which_piece:
                vert_patt_count += 1
            elif state[i][j] != ' ':
                break       
        #Vertical Pattern towards bottom
        i,j = location
        while i < 4:
            i = i + 1
            if state[i][j] == which_piece:
                vert_patt_count += 1
            elif state[i][j] != ' ':
                break


        #diagonal pattern / downward
        i,j = location
        diag1_patt_count = 0  
        while j > 0 and i > 0:
            j = j - 1
            i = i - 1
            if state[i][j] == which_piece:
                diag1_patt_count += 1
            elif state[i][j] != ' ':
                break
        #diagonal pattern / upward
        i,j = location
        while j < 4 and i < 4:
            j = j + 1
            i = i + 1
            if state[i][j] == which_piece:
                diag1_patt_count += 1
            elif state[i][j] != ' ':
                break
        

        #diagonal pattern \ downward
        i,j = location
        diag2_patt_count = 0  
        while j < 4 and i > 0:
            j = j + 1
            i = i - 1
            if state[i][j] == which_piece:
                diag2_patt_count += 1
            elif state[i][j] != ' ':
                break
        #diagonal pattern \ upward
        i,j = location
        while j > 0 and i < 4:
            j = j - 1
            i = i + 1
            if state[i][j] == which_piece:
                diag2_patt_count += 1
            elif state[i][j] != ' ':
                break
        
        
        #3x3 box pattern
        i,j = location
        box_patt_count = -1
        for m in [i-2, i, i + 2]:
            for n in [j - 2, j, j + 2]:
                if m >= 0 and m <= 4 and n >= 0 and n<= 4 and state[m][n] == which_piece:
                    box_patt_count += 1
                elif m >= 0 and m <= 4 and n >= 0 and n<= 4 and state[m][n] != ' ':
                    box_patt_count = box_patt_count - 1
        if box_patt_count < 0:
            box_patt_count = 0

        # return the maximum of all
        return max([box_patt_count, horz_patt_count, vert_patt_count, diag1_patt_count, diag2_patt_count])*0.25 - 0.01


    def heuristic_game_value(self, state):
        '''
         Heuristic function to evaluate non-terminal states. 
        '''
        # check if we have NOT reached the game termination stage
        game_status = self.game_value(state)
        if game_status == 1 or game_status == -1:
            return game_status

        AI_loc_lis = []
        Opp_loc_lis = []
        for i in range(len(state)):
            for j in range(len(state[i])):
                if state[i][j] == self.my_piece:
                    AI_loc_lis.append((i, j))
                if state[i][j] == self.opp:
                    Opp_loc_lis.append((i, j))

        if len(AI_loc_lis) == 0 and len(Opp_loc_lis) == 0:
            return 0 #Game has not started

        # Evaluate pattern match heuristics score for Max and Min Players
        max_pattn_score = 0
        if len(AI_loc_lis) != 0:
            score = []
            for i in range(len(AI_loc_lis)):
                score.append(self.check_pattern(state, AI_loc_lis[i], self.my_piece))
            max_pattn_score = max(score) #Whichever pattern combination is the best

        min_pattn_score = 0
        if len(Opp_loc_lis) != 0:
            score_2 = []
            for i in range(len(Opp_loc_lis)):
                score_2.append(self.check_pattern(state, Opp_loc_lis[i], self.opp))
            min_pattn_score = -1 * max(score_2)

        #Final heuristic: we want AI piece to have better pattern score and undermine the pattern score of opp player
        return max_pattn_score + min_pattn_score


        
    def max_value(self, state, depth):
        """
        Max method to maximize the utilities/reward for Max Player
        """
        if self.game_value(state) != 1:
            pass
        else:
            return self.game_value(state), state
        
        if depth < 3:
            pass
        else:
            return self.heuristic_game_value(state), state
        
        alpha, child = float('-Inf'), state
        #succ_num = 0
        for successor in self.succ(state):
            #succ_num += 1
            prev = alpha
            alpha = max(alpha, self.min_value(successor, depth + 1)[0])
            if alpha <= prev:
                pass
            else:
                child = successor
        return alpha, child

    def min_value(self, state, depth):
        """
        method to minimize the utilities/reward for Min Player
        """
        if self.game_value(state) != -1:
            pass
        else:
            return self.game_value(state), state
        if depth < 3:
            pass
        else:
            return self.heuristic_game_value(state), state
        
        beta, child = float('Inf'), state
        #succ_num = 0
        for successor in self.succ(state):
            #succ_num += 1
            prev = beta
            beta = min(beta, self.max_value(successor, depth + 1)[0])
            if beta >= prev:
                pass
            else:
                child = successor
        return beta, child
    
    
    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this Teeko2Player object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """
        drop_phase = True 

        # DONE: detect completion of drop phase 
        count = 0
        for row in state:
            for item in row:
                if item != ' ':
                    count += 1
        if count >= 8:
            drop_phase = False

        #Drop phase
        move = []
        if drop_phase:
            max_val, next_state = self.max_value(state, 0)
            for i in range(5):
                for j in range(5):  
                    if state[i][j] == ' ' and next_state[i][j] == self.my_piece:
                        move.insert(0, (i, j))
            return move       

        #Continued Gameplay: choose a piece to move and remove it from the board
        max_val, next_state = self.max_value(state, 0)
        for i in range(5):
            for j in range(5):
                if state[i][j] == ' ' and next_state[i][j] == self.my_piece:
                    move.insert(0, (i, j))
                if state[i][j] == self.my_piece and next_state[i][j] == ' ':
                    move.insert(1, (i, j))
        return move  
        
    
    def opponent_make_move(self, state):
        """Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move       """

        #drop_phase = is_drop_phase(self.state)   # TODO: detect drop phase
        count = 0
        for row in state:
            for item in row:
                if item != ' ':
                    count += 1
        if count == 8:
            drop_phase = False
        else:
            drop_phase = True 

        opp_loc = []
        opp_next_loc = []
        move = []
        
        if not drop_phase:
            # TODO: choose a piece to move and remove it from the board
            for i in range(5):
                for j in range(5):
                    if state[i][j] == self.opp:
                        opp_loc.append((i,j))
            rand_loc_choose = random.randint(0,len(opp_loc)-1)
            (i,j) = opp_loc[rand_loc_choose]
            adj_locations = [[i, j + 1], [i, j - 1], [i + 1, j], [i - 1, j], [i + 1, j + 1], [i + 1, j - 1], [i - 1, j + 1],
                         [i - 1, j - 1]]
                # iterate over all possible directions and check if it is still within the board
            for next_location in adj_locations:
                if 0 <= next_location[0] and next_location[0] < 5 and 0 <= next_location[1] and next_location[1] < 5 and state[next_location[0]][next_location[1]] == ' ':
                    opp_next_loc.append(next_location)
            if len(opp_next_loc)==1:
                random_choose_move = opp_next_loc[0]
            else:
                random_choose_move  = opp_next_loc[random.randint(0,len(opp_next_loc)-1)]
            move.append(random_choose_move)
            move.append((i,j))
            return move
                    
            

        # select an unoccupied space randomly
        # TODO: implement a minimax algorithm to play better
        move = []
        (row, col) = (random.randint(0,4), random.randint(0,4))
        while not state[row][col] == ' ':
            (row, col) = (random.randint(0,4), random.randint(0,4))

        # ensure the destination (row,col) tuple is at the beginning of the move list
        move.insert(0, (row, col))
        return move
    
    
    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this Teeko2Player object, or a generated successor state.

        Returns:
            int: 1 if this Teeko2Player wins, -1 if the opponent wins, 0 if no winner

        DONE: complete checks for diagonal and 3x3 square corners wins
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i]==self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col]==self.my_piece else -1

        # DONE: check \ diagonal wins
        for x in range(2):
            for y in range(2):
                if state[x][y] != ' ' and state[x][y] == state[x+1][y+1] == state[x+2][y+2] == state[x+3][y+3]:
                    return 1 if state[x][y]==self.my_piece else -1
                
        # DONE: check / diagonal wins
        for x in range(2):
            for y in range(3, 5):
                if state[x][y] != ' ' and state[x][y] == state[x+1][y-1] == state[x+2][y-2] == state[x+3][y-3]:
                    return 1 if state[x][y] == self.my_piece else -1

        # DONE: check 3x3 square corners wins
        for x in range(3):
            for y in range(3):
                if state[x][y] == ' ' and state[x-1][y-1] != ' ' and state[x-1][y-1] == state[x-1][y+1] == state[x+1][y-1] == state[x+1][y+1]:
                    return 1 if state[x+1][y+1] == self.my_piece else -1

        return 0 # no winner yet


############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY  (MANUAL PLAY)
#
############################################################################
"""
def main1():
    print('Hello, this is Samaritan')
    ai = Teeko2Player()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            t1 = time.time()
            move = ai.make_move(ai.board)
            t2 = time.time()
            print(t2-t1)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            t1 = time.time()
            move = ai.make_move(ai.board)
            t2 = time.time()
            print(t2-t1)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")
"""
############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY  (RAMDOM PLAY)
#
############################################################################        
        
def main():
    print('Hello, this is Samaritan')
    ai = Teeko2Player()
    piece_count = 0
    turn = 0
    print("AI: "+ ai.my_piece)
    #print("Opp: "+ ai.opp)
    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            #ai.print_board()
            t1 = time.time()
            move = ai.make_move(ai.board)
            t2 = time.time()
            print(t2-t1)
            ai.place_piece(move, ai.my_piece)
            #print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            #ai.print_board()
            #print(ai.opp+"'s turn")
            move = ai.opponent_make_move(ai.board)
            ai.place_piece(move, ai.opp)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            #ai.print_board()
            t1 = time.time()
            move = ai.make_move(ai.board)
            t2 = time.time()
            print(t2-t1)
            #print("AI move after drop")
            #print(move)
            ai.place_piece(move, ai.my_piece)
            #print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            #print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            #move_made = False
            #ai.print_board()
            move = ai.opponent_make_move(ai.board)
            #print("Opp move after drop")
            #print(move)
            ai.place_piece(move, ai.opp)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    win_count = 0
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
        win_count +=1
    else:
        print("You win! Game over.")

if __name__ == "__main__":
    main()
