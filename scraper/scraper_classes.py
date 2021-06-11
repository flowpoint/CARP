class StoryMetadata:
    def __init__(self, story_title, story_link, author, author_link, word_count, genre, crit_count):
        self.story_title = story_title
        self.story_link = story_link
        self.author = author
        self.author_link = author_link
        self.word_count = word_count
        self.genre = genre
        self.crit_count = crit_count

class RowStoryMetadata:
    def __init__(self, story_title, story_link, author, author_link, word_count, genre, crit_count):
        self.story_title = story_title
        self.story_link = story_link
        self.author = author
        self.author_link = author_link
        self.word_count = word_count
        self.genre = genre
        self.crit_count = crit_count

class FullStoryMetadata:
    def __init__(self, story_id, story_title, story_link, author, author_link, word_count, genre, crit_count, author_notes, story_text):
        self.story_id = story_id
        self.story_title = story_title
        self.story_link = story_link
        self.author = author
        self.author_link = author_link
        self.word_count = word_count
        self.genre = genre
        self.crit_count = crit_count
        self.author_notes = author_notes
        self.story_text = story_text

class StoredMetadata:
    def __init__(self, story_id, story_title, story_link, author, author_link, word_count, genre, crit_count, author_notes, story_text, is_processed):
        self.story_id = story_id
        self.story_title = story_title
        self.story_link = story_link
        self.author = author
        self.author_link = author_link
        self.word_count = word_count
        self.genre = genre
        self.crit_count = crit_count
        self.author_notes = author_notes
        self.story_text = story_text
        self.is_processed = is_processed

class CritiqueMetadata:
    def __init__(self, submission_id, critic_name, critic_link, critique_link, word_count, critique_type):
        self.submission_id = submission_id
        self.critic_name = critic_name
        self.critic_link = critic_link
        self.critique_link = critique_link
        self.word_count = word_count
        self.critique_type = critique_type

class FullCritique:
    def __init__(self, submission_id, critic_name, critic_link, critique_link, word_count, critique_type, story_target, target_comment):
        self.submission_id = submission_id
        self.critic_name = critic_name
        self.critic_link = critic_link
        self.critique_link = critique_link
        self.word_count = word_count
        self.critique_type = critique_type
        self.story_target = story_target
        self.target_comment = target_comment

class StoredCritique:
    def __init__(self, critique_id, submission_id, critic_name, critic_link, critique_link, word_count, critique_type, story_target, target_comment):
        self.critique_id = critique_id
        self.submission_id = submission_id
        self.critic_name = critic_name
        self.critic_link = critic_link
        self.critique_link = critique_link
        self.word_count = word_count
        self.critique_type = critique_type
        self.story_target = story_target
        self.target_comment = target_comment