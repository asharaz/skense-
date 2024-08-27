

# Path to the hosts file (adjust path for your OS)
hosts_file_path = r'C:\Windows\System32\drivers\etc\hosts'  # For Windows


# Redirect IP address
redirect = '127.0.0.1'

# Input from the user
action = input("Do you want to block or unblock a website? (block/unblock): ").strip().lower()
website = input("Enter the website (e.g., www.example.com): ").strip()

# Open the hosts file and modify the entry
with open(hosts_file_path, "r+") as hosts_file:
    # Read the current content of the file
    content = hosts_file.readlines()
    hosts_file.seek(0)  # Move to the start of the file

    # Prepare the entry to block the website
    entry = f"{redirect} {website}\n"

    if action == 'block':
        # Add the website to the hosts file if not already blocked
        if entry not in content:
            hosts_file.write(entry)
            print(f"Blocked: {website}")
        else:
            print(f"Already blocked: {website}")

    elif action == 'unblock':
        # Remove the website from the hosts file if it is blocked
        for line in content:
            if line.strip() != entry.strip():
                hosts_file.write(line)
        print(f"Unblocked: {website}")

    else:
        print("Invalid action. Please choose 'block' or 'unblock'.")

    hosts_file.truncate()  # Truncate file to remove any old data beyond the current cursor position
