import spacy
from datetime import datetime, timedelta
import re

class EmailQueryProcessor:
    def __init__(self, gmail_reader):
        self.gmail_reader = gmail_reader
        self.nlp = spacy.load("en_core_web_sm")
        self.system_prompt_for_analysis = """
        You are an AI assistant designed to analyze a list of emails and provide a concise summary,
        highlighting the most urgent or important emails based on the user's original intent.
        The user's original query was: {original_query}

        Here is the list of emails:
        {emails_json}

        Please provide:
        1. A brief summary of the emails.
        2. A list of the most urgent/important emails, explaining why they are urgent.
        3. Any other relevant insights.
        """

    def _parse_query_with_nlp(self, query_text):
        doc = self.nlp(query_text.lower())
        params = {
            "read_status": None,
            "sender": None,
            "subject": None,
            "start_date": None,
            "end_date": None,
            "max_results": None
        }

        # Read status
        if "unread" in doc.text:
            params["read_status"] = "unread"
        elif "read" in doc.text:
            params["read_status"] = "read"

        # Sender
        for ent in doc.ents:
            if ent.label_ == "PERSON" or ent.label_ == "ORG":
                if "from" in ent.sent.text:
                    params["sender"] = ent.text
                    break
        
        # Subject (simple keyword extraction for now)
        subject_keywords = []
        if "about" in doc.text:
            # Extract text after "about" until a known parameter keyword or end of query
            match = re.search(r'about\s+(.*?)(?:from|read|unread|past|last|today|yesterday|\d+)?', doc.text)
            if match:
                subject_keywords.append(match.group(1).strip())
        if subject_keywords:
            params["subject"] = " ".join(subject_keywords)

        # Max results
        for token in doc:
            if token.like_num:
                if "recent" in token.sent.text or "latest" in token.sent.text or "newest" in token.sent.text:
                    params["max_results"] = int(token.text)
                    break

        # Dates
        # This part will rely on _convert_relative_dates for interpretation
        if "yesterday" in doc.text:
            params["start_date"] = "yesterday"
            params["end_date"] = "yesterday"
        elif "past 7 days" in doc.text or "last week" in doc.text:
            params["start_date"] = "past 7 days"
            params["end_date"] = "today"
        elif "today" in doc.text:
            params["start_date"] = "today"
            params["end_date"] = "today"
        # Add more date parsing as needed, e.g., specific dates using spaCy's date entities

        return params

    def _convert_relative_dates(self, params):
        today = datetime.now()
        
        if "start_date" in params and params["start_date"]:
            if "yesterday" in params["start_date"]:
                params["start_date"] = (today - timedelta(days=1)).strftime('%Y/%m/%d')
                if "end_date" not in params or not params["end_date"]:
                    params["end_date"] = params["start_date"] # If only yesterday is mentioned, assume it's for yesterday only
            elif "past 7 days" in params["start_date"] or "last week" in params["start_date"]:
                params["start_date"] = (today - timedelta(days=7)).strftime('%Y/%m/%d')
                if "end_date" not in params or not params["end_date"]:
                    params["end_date"] = today.strftime('%Y/%m/%d')
            # Add more relative date conversions as needed
        
        if "end_date" in params and params["end_date"]:
            if "today" in params["end_date"]:
                params["end_date"] = today.strftime('%Y/%m/%d')
            # Add more relative date conversions as needed

        return params

    def process_email_query(self, query_text, gemini_model):
        # Step 1: Parse query to extract parameters using spaCy
        extracted_params = self._parse_query_with_nlp(query_text)
        
        # Step 2: Convert relative dates to absolute dates
        processed_params = self._convert_relative_dates(extracted_params)

        # Step 3: Call GmailReader with extracted parameters
        read_status = processed_params.get("read_status")
        sender = processed_params.get("sender")
        subject = processed_params.get("subject")
        start_date = processed_params.get("start_date")
        end_date = processed_params.get("end_date")
        max_results = processed_params.get("max_results")
        if max_results is None:
            max_results = 10 # Default to 10 if not specified in the query

        emails = self.gmail_reader.list_emails(
            read_status=read_status,
            sender=sender,
            subject=subject,
            start_date=start_date,
            end_date=end_date,
            max_results=max_results
        )

        if not emails:
            return "No emails found matching your criteria."

        # Step 4: Analyze emails for urgency/summary using Gemini (LLM)
        emails_json = str(emails) # Convert list of dicts to string for prompt
        analysis_prompt = self.system_prompt_for_analysis.format(
            original_query=query_text,
            emails_json=emails_json
        )
        try:
            analysis_response = gemini_model.generate_content(analysis_prompt)
            return analysis_response.text
        except Exception as e:
            return f"An error occurred during email analysis: {e}"