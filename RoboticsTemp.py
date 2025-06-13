import turtle

Roboto = turtle.Turtle()

Roboto.shape("turtle")
Roboto.speed(2)


def go_Forward():
    Roboto.forward(20)


def go_Backward():
    Roboto.backward(20)

def Go_left():
    Roboto.left(20)


def go_Right():
    Roboto.right(20)


screen = turtle.Screen()

screen.title("The roboto")

screen.listen()

screen.onkey(go_Forward, "Up")
screen.onkey(go_Backward, "Down")
screen.onkey(go_Right,"Right")
screen.onkey(Go_left,"Left")

turtle.done()