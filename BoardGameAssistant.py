import cv2
import json
from tgbot import tgbot
from gameClassifier import gameClassifier
from checkersDetector import checkersDetector

detector = checkersDetector.CheckersDetector(pathToYolo='checkersDetector/yolov5', pathToModel='checkersDetector/detector.pt')

def detect_checkers(image_path, chat_id):
	img = cv2.imread(image_path)
	res = detector.getGameField(img,visualize=True)
	cv2.imwrite("layout.jpg", res[1])
	cv2.imwrite("game.jpg", res[0])
	bot.send_message(chat_id=chat_id, text='Detected layout:')
	bot.send_photo(chat_id=chat_id, photo=open("layout.jpg", 'rb'))
	bot.send_message(chat_id=chat_id, text='Digitized game:')
	bot.send_photo(chat_id=chat_id, photo=open("game.jpg", 'rb'))

CONFIG_PATH = "config.json"
with open(CONFIG_PATH) as f:
	config = json.load(f)

BOT_TOKEN = config['TG_BOT_KEY']
bot = tgbot.initBot(token=BOT_TOKEN, classifyGameFunction=gameClassifier.classifyGameImage, detect_checkersFunction=detect_checkers)
