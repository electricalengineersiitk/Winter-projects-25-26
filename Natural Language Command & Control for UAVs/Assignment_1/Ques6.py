class Book:
    def __init__(self, title, author):
        self.title = title
        self.author = author
        self.is_checked_out = False

class Library:
    def __init__(self):
        self.books = []

    def add_book(self, book_obj):
        self.books.append(book_obj)

    def checkout_book(self, title):
        for book in self.books:
            if book.title == title:
                if book.is_checked_out:
                    print(f"Error: '{title}' is already checked out.") 
                else:
                    book.is_checked_out = True
                    print(f"Success: You checked out '{title}'.")
                return
        print("Error: Book not found.")

    def return_book(self, title):
        for book in self.books:
            if book.title == title:
                book.is_checked_out = False 
                print(f"Success: Returned '{title}'.")
                return
        print("Error: Book not found in library records.")