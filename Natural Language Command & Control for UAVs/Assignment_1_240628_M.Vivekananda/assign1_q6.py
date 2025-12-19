class Book:
    def __init__(self,title,author):
        self.title=title
        self.author=author
        self.checked_out=False


class Library:
    def __init__(self):
        self.books=[]

    def adding_book(self,book):
        self.books.append(book)
        print("Book added")
    
    def checkout_book(self,title):
        for i in self.books:
            if i.title==title:
                if i.checked_out:
                    print("Book already checked out bro")
                    return
                else:
                    i.checked_out=True
                    print("Checked our successfullu")
                    return
        print("No book with this name")
    
    def returning_book(self,title):
        for i in self.books:
            if i.title==title:
                if not i.checked_out:
                    print("Book is not checker out")
                    return
                else:
                    i.is_checked_out=False
                    print("Returned successfullu")
                    return
        print("Not found")


lib=Library()
book_1=Book("Dnjfndo","kjfheoiue")
book_2=Book("djbhdib","idhfpihfuw")
book_3=Book("ncdkcnoien","idhfpihfuw")

lib.adding_book(book_1)
lib.adding_book(book_2)
lib.adding_book(book_3)


lib.checkout_book("Dnjfndo")
lib.checkout_book("ncdkcnoien")

lib.returning_book("Dnjfndo")
lib.checkout_book("CIDJNUINIDCDIN")