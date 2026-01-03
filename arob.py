if __name__ == "__main__":
    # Simple CLI for Demo
    game = HigherLowerGame()
    print("="*40)
    print("   HIGHER OR LOWER - CONSOLE DEMO")
    print("="*40)
    
    while not game.is_game_over:
        print(f"\n[SCORE]: {game.score}")
        print(f"[CARD] : {game.current_card}")
        print("-" * 20)
        
        choice = input("Next card Higher (h) or Lower (l)? ").lower().strip()
        
        if choice not in ['h', 'l', 'higher', 'lower']:
            print("Invalid input! Type 'h' or 'l'.")
            continue
            
        guess = "higher" if choice in ['h', 'higher'] else "lower"
        result = game.check_guess(guess)
        
        print(f"\n>>> {game.message}")
        if not result:
            # Optional: End game on wrong guess for demo thrill
            # game.is_game_over = True
            pass
            
    print("\nGAME OVER")
    print(f"Final Score: {game.score}")