# Website Blocker and Unblocker

## Project Background

This project provides a simple Python script to manage access to specific websites on your local machine. 
By modifying the system's `hosts` file, the script allows users to block or unblock websites. Blocking a 
website will redirect any attempt to access it to `127.0.0.1`, effectively preventing the site from loading 
in any web browser on the system.

This tool can be particularly useful for parental control, productivity enhancement, or restricting access 
to unwanted content. It is designed to work on Windows, with a path provided for macOS/Linux systems as well.

## Features

- **Block Websites:** Easily block access to specified websites by adding entries to the system's `hosts` file.
- **Unblock Websites:** Remove block entries from the `hosts` file, restoring access to previously blocked websites.
- **Simple Interface:** The script runs in a command-line interface, prompting the user to block or unblock a website 
by entering the URL.

## Prerequisites

- **Operating System:** Windows, macOS, or Linux.
- **Administrator Privileges:** Modifying the `hosts` file requires administrator rights, so the script must be run 
with elevated privileges.

## How to Use

1. **Run the Script:**
   - Execute the Python script in an environment where you have administrator privileges.

2. **Choose an Action:**
   - The script will prompt you to choose whether you want to `block` or `unblock` a website.
   - Enter `block` to block a website or `unblock` to remove the block.

3. **Enter the Website:**
   - Input the full website address you wish to block or unblock (e.g., `www.example.com`).

4. **Confirmation:**
   - The script will confirm whether the website has been successfully blocked or unblocked.

## Code Explanation

### Define the Path to the Hosts File
```
hosts_file_path = r'C:\Windows\System32\drivers\etc\hosts'  # For Windows
redirect = '127.0.0.1'
```
* hosts_file_path: Specifies the file path to the hosts file, which is used to map hostnames (like domain names) to IP addresses on the local machine. The r before the string indicates a raw string, meaning that backslashes are treated as literal characters. This path is specific to Windows. On other operating systems like Linux or macOS, the path would differ. Define the Redirect IP Address
* redirect: Specifies the IP address to which the website should be redirected when blocking it. 127.0.0.1 is the IP address for the local machine (localhost). Redirecting a website to this address effectively blocks access to it, as the request will loop back to the user's own computer instead of reaching the intended website.
### Get User Input
```
action = input("Do you want to block or unblock a website? (block/unblock): ").strip().lower()
website = input("Enter the website (e.g., www.example.com): ").strip()
```
* Purpose: Captures user input for the desired action (block or unblock) and the website to be affected.The .strip() method removes any leading or trailing whitespace from the input, and .lower() converts the action to lowercase, making the input case-insensitive.
### Open the Hosts File and Modify the Entry
```
with open(hosts_file_path, "r+") as hosts_file:
```
* Purpose: Opens the hosts file for both reading and writing (r+ mode). The with statement ensures that the file is properly closed after the block of code is executed, even if an error occurs.
### Read the Current Content of the File
```
content = hosts_file.readlines()
hosts_file.seek(0)  # Move to the start of the file
```
* Purpose: Reads all the lines in the file into a list (content) and then moves the file cursor back to the start. readlines() reads the entire file line by line, storing each line as an element in the content list. seek(0) resets the cursor to the beginning of the file, preparing it for the next operation.
### Prepare the Entry to Block the Website
````
entry = f"{redirect} {website}\n"
````
* Purpose: Formats the entry that will be added to or removed from the hosts file. This line creates a string in the format "127.0.0.1 www.example.com\n", which is how entries are typically structured in the hosts file to block a website.
### Handle Blocking the Website
````
if action == 'block':
````
* Purpose: Checks if the user wants to block the website. If the user input for action is 'block', the code inside this block will execute.
### Add the Website to the Hosts File if Not Already Blocked
````
if entry not in content:
    hosts_file.write(entry)
    print(f"Blocked: {website}")
else:
    print(f"Already blocked: {website}")
````
* Purpose: Adds the entry to the hosts file if it doesn’t already exist.The code checks if the entry is not already in content. If it's not, the entry is added to the file, and a confirmation message is printed. If it is already there, it informs the user that the website is already blocked.
### Handle Unblocking the Website
````
elif action == 'unblock':
````
* Purpose: Checks if the user wants to unblock the website. If the user input for action is 'unblock', the code inside this block will execute.

### Remove the Website from the Hosts File if Blocked
````
for line in content:
    if line.strip() != entry.strip():
        hosts_file.write(line)
print(f"Unblocked: {website}")
````
* Purpose: Removes the blocking entry from the hosts file. The code loops through each line in content. If a line matches the entry, it is not written back to the file, effectively removing it. A message is then printed to confirm the unblocking.

### Handle Invalid Actions
````
else:
    print("Invalid action. Please choose 'block' or 'unblock'.")
````
* Purpose: Handles any input other than 'block' or 'unblock'. If the user enters an invalid action, this block will notify them and provide correct options.

### Truncate the File
````
hosts_file.truncate()
````
* Purpose: Removes any remaining data in the file beyond the current cursor position. After rewriting the necessary lines in the hosts file, truncate() ensures that any leftover lines from the original content are removed.
## Output

We first run the code to block/unblock any website.
![Screenshot 2024-08-26 112009](https://github.com/user-attachments/assets/7c54a629-238f-4650-9ef9-cf1c3535db7d)



we need to input the required option.Enter the webiste that need to be blocked.
![Screenshot 2024-08-26 112044](https://github.com/user-attachments/assets/6effa785-6f33-49ad-b8be-59b50b548b89)


the enetered url is blocked and if you enter the webiste url it wont connect to the server.
![Screenshot 2024-08-26 112103](https://github.com/user-attachments/assets/46b235ae-2f5a-4c59-bfea-29cd8c1cdcb7)


Now to unblock follow the same procedure but enter unblock.
![Screenshot 2024-08-26 112552](https://github.com/user-attachments/assets/6e0aae02-9ec3-4e9a-bcb4-0a17cb32f3a6)


The website is now unblocked and it connects to the server.
![Screenshot 2024-08-26 112626](https://github.com/user-attachments/assets/88494320-9689-494c-b592-852d0e90af34)





