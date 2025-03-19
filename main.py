import train_test as tt

def main():
    is_continue = "y"
    
    while is_continue == "y":
        carat = float(input("Enter the carat : "))
        pred_price = tt.w * carat + tt.b
        if tt.r2 > 0.7 :
            print(f"Predicted price is : {pred_price} with high predictive power")
        elif tt.r2 > 0.5 :
            print(f"Predicted price is : {pred_price} with moderate predictive power")
        else:
            print(f"Predicted price is : {pred_price} with low predictive power")

        is_continue = input("Do you want to continue? (y/n)")

if __name__ == "__main__":
    main()
