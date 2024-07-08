import streamlit as st
import mysql.connector
import re
import base64

# Function to check if the username is valid
def is_valid_username(username):
    # Username should contain at least one number
    return any(char.isdigit() for char in username)

# Function to check if the password is valid
def is_valid_password(password):
    # Password should contain at least 1 capital letter, 1 number, and be at least 8 characters long
    if re.search(r"^(?=.*[A-Z])(?=.*\d).{8,}$", password):
        return True
    return False

# Function to check if the passwords match
def passwords_match(password, confirm_password):
    return password == confirm_password

# Function to register a new user
def register_user(username, email, password):
    try:
        # Establish connection to MySQL database
        connection = mysql.connector.connect(
            host="localhost",
            port=3306,
            user="root",  # Replace with your MySQL username
            password="root",  # Replace with your MySQL password
            database="userdetails"  # Replace with your MySQL database name
        )
        cursor = connection.cursor()

        # Prepare SQL query to insert user data into the database
        sql = "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)"
        user_data = (username, email, password)

        # Execute the SQL query
        cursor.execute(sql, user_data)

        # Commit the transaction
        connection.commit()

        # Print registration success message
        print("Registration successful! You can now log in.")
        st.success("Registration successful! You can now log in.")

    except mysql.connector.Error as error:
        # Handle any errors that occur during registration
        print("Error registering user:", error)
        st.error(f"Error registering user: {error}")

    finally:
        # Close the cursor and connection
        cursor.close()
        connection.close()

# Function to set background image
def add_bg_from_local(image_file):
    try:
        with open(image_file, "rb") as file:
            encoded_string = base64.b64encode(file.read()).decode()
            st.markdown(
                f"""
                <style>
                .stApp {{
                    background-image: url('data:image/png;base64,{encoded_string}');
                    background-size: cover;
                }}
                </style>
                """,
                unsafe_allow_html=True
            )
    except FileNotFoundError:
        st.error(f"Error: Image file '{image_file}' not found.")

# Call the function to set background image
add_bg_from_local('1.jpg')

# Title
st.markdown('<h1 style="color:#FFFFFF;text-align: center;font-size:36px;">Registration Page</h1>', unsafe_allow_html=True)

# Input fields for registration
username = st.text_input("Username (should contain at least one number)")
email = st.text_input("Email")
password = st.text_input("Password", type="password")
confirm_password = st.text_input("Confirm Password", type="password")

# Register button
if st.button("Register"):
    # Validate username
    if not is_valid_username(username):
        st.error("Username should contain at least one number!")
    # Validate password
    elif not is_valid_password(password):
        st.error("Password should contain at least 1 capital letter, 1 number, and be at least 8 characters long!")
    # Check if passwords match
    elif not passwords_match(password, confirm_password):
        st.error("Passwords do not match!")
    else:
        # Registration successful
        register_user(username, email, password)

# Login button to redirect to login page


if st.button("Login"):
    # Redirect to login page
    import subprocess
    subprocess.run(['streamlit', 'run', 'login.py'])