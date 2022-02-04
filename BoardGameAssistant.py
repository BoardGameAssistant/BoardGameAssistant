import cv2
import json
from tgbot import tgbot
from tictactoe import pipeline
from gameClassifier import gameClassifier
from checkers import checkersDetector

detector = checkersDetector.CheckersDetector(pathToYolo='checkers/yolov5', pathToModel='checkers/models/detector.pt')

def detect_checkers(image_path, chat_id, top_white):
	img = cv2.imread(image_path)
	visual, layout, res_white, res_black  = detector.getGameField(img,visualize=True, roll=top_white)
	cv2.imwrite("layout.jpg", layout)
	cv2.imwrite("game.jpg", visual)
	cv2.imwrite("res_white.jpg", res_white)
	cv2.imwrite("res_black.jpg", res_black)
	bot.send_message(chat_id=chat_id, text='Detected layout:')
	bot.send_photo(chat_id=chat_id, photo=open("layout.jpg", 'rb'))
	bot.send_message(chat_id=chat_id, text='Digitized game:')
	bot.send_photo(chat_id=chat_id, photo=open("game.jpg", 'rb'))
	bot.send_message(chat_id=chat_id, text='Suggested white move:')
	bot.send_photo(chat_id=chat_id, photo=open("res_white.jpg", 'rb'))
	bot.send_message(chat_id=chat_id, text='Suggested black move:')
	bot.send_photo(chat_id=chat_id, photo=open("res_black.jpg", 'rb'))

def detect_tictactoe(image_path, chat_id):
	result = pipeline.execute_pipline(image_path=image_path)
	print(len(result))
	if result[0] == 'Cannot detect game properly':
		bot.send_message(chat_id=chat_id, text='Unable to recognize game situation')
	else:
		cv2.imwrite("ttt.jpg", result[0])
		bot.send_photo(chat_id=chat_id, photo=open("ttt.jpg", 'rb'))

CONFIG_PATH = "config.json"
with open(CONFIG_PATH) as f:
	config = json.load(f)

BOT_TOKEN = config['TG_BOT_KEY']

handlers_fns = {
	'classifyGame': gameClassifier.classifyGameImage, 
	'detect_checkers': detect_checkers,
	'detect_tictactoe': detect_tictactoe
}
bot = tgbot.initBot(token=BOT_TOKEN, handlers_fns=handlers_fns)
