
'''''
import streamlit as st
import mysql.connector
import bcrypt

# Function to authenticate user
def authenticate_user(email, password):
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

        # Prepare SQL query to fetch user data from the database using email
        sql = "SELECT * FROM users WHERE email = %s"
        user_data = (email,)

        # Execute the SQL query
        cursor.execute(sql, user_data)

        # Fetch the result
        result = cursor.fetchone()

        if result:
            # Extract the stored hashed password
            stored_hashed_password = result[2]  # Assuming password hash is stored in the third column

            # Check if the provided password matches the stored hashed password
            if bcrypt.checkpw(password.encode('utf-8'), stored_hashed_password.encode('utf-8')):
                return True  # Authentication successful
        return False  # Authentication failed

    except mysql.connector.Error as error:
        # Handle any errors that occur during authentication
        print("Error authenticating user:", error)
        st.error(f"Error authenticating user: {error}")

    finally:
        # Close the cursor and connection
        cursor.close()
        connection.close()

# Title
st.markdown('<h1 style="color:#FFFFFF;text-align: center;font-size:36px;">Login Page</h1>', unsafe_allow_html=True)

# Input fields for login
email = st.text_input("Email")
password = st.text_input("Password", type="password")

# Login button
if st.button("Login"):
    # Authenticate user using email and bcrypt
    if authenticate_user(email, password):
        st.success("Login successful! Welcome, " + email)
        # Redirect to Fire.py page or perform further actions upon successful login
        import subprocess
        subprocess.run(['streamlit', 'run', 'Fire.py'])
    else:
        st.error("Invalid email or password. Please try again.")
'''
import streamlit as st
import mysql.connector
import subprocess
import base64

# Connect to MySQL database
conn = mysql.connector.connect(
    host="localhost",
    port=3306,
    user="user ",
    password="password",
    database="databse name"
)

# Function to add a background image to Streamlit app
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

# Define Streamlit app
def main():
    # Set background image
    add_bg_from_local('1.jpg')

    st.title("Login Page")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        # Authenticate user
        if authenticate(email, password):
            st.success("Logged in successfully!")
            # Redirect to Fire.py page
            subprocess.run(['streamlit', 'run', 'Fire.py'])
        else:
            st.error("Invalid email or password.")

def authenticate(email, password):
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE email = %s AND password = %s"
    cursor.execute(query, (email, password))
    user = cursor.fetchone()
    cursor.fetchall()  # Fetch all remaining results
    cursor.close()
    if user:
        return True
    else:
        return False


if __name__ == "__main__":
    main()
