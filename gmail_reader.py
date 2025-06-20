import imaplib
import email
from datetime import datetime

class GmailReader:
    def __init__(self, email_address, app_password):
        self.email_address = email_address
        self.app_password = app_password
        self.mail = self._authenticate_gmail()

    def _authenticate_gmail(self):
        try:
            mail = imaplib.IMAP4_SSL('imap.gmail.com')
            mail.login(self.email_address, self.app_password)
            print("Successfully authenticated with Gmail IMAP.")
            return mail
        except Exception as e:
            print(f"Error authenticating with Gmail IMAP: {e}")
            return None

    def list_emails(self, read_status=None, sender=None, subject=None, start_date=None, end_date=None, max_results=10):
        if not self.mail:
            print("Gmail IMAP service not available. Cannot list emails.")
            return []

        try:
            self.mail.select('inbox')
            
            search_criteria = []
            if read_status == "read":
                search_criteria.append('SEEN')
            elif read_status == "unread":
                search_criteria.append('UNSEEN')
            else:
                search_criteria.append('ALL')

            if sender:
                search_criteria.append(f'FROM "{sender}"')
            
            if subject:
                search_criteria.append(f'SUBJECT "{subject}"')
            
            if start_date:
                try:
                    date_obj = datetime.strptime(start_date, '%Y/%m/%d')
                    search_criteria.append(f'SENTSINCE "{date_obj.strftime("%d-%b-%Y")}"')
                except ValueError:
                    print("Invalid start date format. Please use YYYY/MM/DD.")
            
            if end_date:
                try:
                    date_obj = datetime.strptime(end_date, '%Y/%m/%d')
                    search_criteria.append(f'SENTBEFORE "{date_obj.strftime("%d-%b-%Y")}"')
                except ValueError:
                    print("Invalid end date format. Please use YYYY/MM/DD.")

            status, email_ids = self.mail.search(None, *search_criteria)
            email_id_list = email_ids[0].split()
            
            emails_data = []
            for i, email_id in enumerate(reversed(email_id_list)): # Get most recent first
                print(f"Processing email ID: {email_id}, index: {i}")
                print(f"Email ID type: {type(email_id)}")
                if i >= max_results:
                    break
                
                status, msg_data = self.mail.fetch(email_id, '(RFC822)')
                raw_email = msg_data[0][1]
                msg = email.message_from_bytes(raw_email)
                
                email_data = {
                    'id': email_id.decode(),
                    'sender': msg['from'],
                    'subject': msg['subject'],
                    'date': msg['date'],
                    'snippet': self._get_email_snippet(msg)
                }
                emails_data.append(email_data)
            
            return emails_data
        except Exception as e:
            print(f"An error occurred while listing emails: {e}")
            return []

    def _get_email_snippet(self, msg):
        """Extracts a snippet from the email body."""
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                cdispo = str(part.get('Content-Disposition'))

                # Look for plain text parts, not attachments
                if ctype == 'text/plain' and 'attachment' not in cdispo:
                    try:
                        payload = part.get_payload(decode=True).decode()
                        return payload[:200] + "..." if len(payload) > 200 else payload
                    except:
                        return "Could not decode email body."
        else:
            try:
                payload = msg.get_payload(decode=True).decode()
                return payload[:200] + "..." if len(payload) > 200 else payload
            except:
                return "Could not decode email body."
        return "No plain text body found."