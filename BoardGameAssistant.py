import cv2
import json
from PIL import Image
from tgbot import tgbot
from gameClassifier import gameClassifier
from tictactoe import pipeline
from checkers import checkersDetector
from carcassone import tiles_board


detector = checkersDetector.CheckersDetector(pathToYolo='checkers/yolov5', pathToModel='checkers/models/detector.pt')


def detect_tictactoe(image_path, chat_id):
	result = pipeline.execute_pipline(image_path=image_path)
	print(len(result))
	if result[0] == 'Cannot detect game properly':
		bot.send_message(chat_id=chat_id, text='Unable to recognize game situation')
	else:
		cv2.imwrite("ttt.jpg", result[0])
		bot.send_photo(chat_id=chat_id, photo=open("ttt.jpg", 'rb'))


def detect_checkers(image_path, chat_id, top_white):
	img = cv2.imread(image_path)
	visual, layout, res_white, res_black  = detector.getGameField(img,visualize=True, roll=top_white)

	layout_img_path = image_path.replace('image', 'checkers_layout').replace('.png', '.jpg')
	cv2.imwrite(layout_img_path, layout)
	bot.send_message(chat_id=chat_id, text='Detected layout:')
	bot.send_photo(chat_id=chat_id, photo=open(layout_img_path, 'rb'))

	game_img_path = image_path.replace('image', 'checkers_game').replace('.png', '.jpg')
	cv2.imwrite(game_img_path, visual)
	bot.send_message(chat_id=chat_id, text='Digitized game:')
	bot.send_photo(chat_id=chat_id, photo=open(game_img_path, 'rb'))

	res_white_img_path = image_path.replace('image', 'checkers_white').replace('.png', '.jpg')
	cv2.imwrite(res_white_img_path, res_white)
	bot.send_message(chat_id=chat_id, text='Suggested white move:')
	bot.send_photo(chat_id=chat_id, photo=open(res_white_img_path, 'rb'))
	
	res_black_img_path = image_path.replace('image', 'checkers_black').replace('.png', '.jpg')
	cv2.imwrite(res_black_img_path, res_black)
	bot.send_message(chat_id=chat_id, text='Suggested black move:')
	bot.send_photo(chat_id=chat_id, photo=open(res_black_img_path, 'rb'))


def detect_carcassone(field_image_path, card_image_path, chat_id):
	field_image = Image.open(field_image_path)
	card_image = Image.open(card_image_path)

	board = tiles_board.CarcassoneBoard('carcassone/models/detection_model.pt', 'carcassone/models/cls_model.pt')
	board_img = board.recognize_game_situation(field_image)

	board_img_path = field_image_path.replace('image', 'board_img').replace('.png', '.jpg')
	cv2.imwrite(board_img_path, board_img)
	bot.send_message(chat_id=chat_id, text='Detected board:')
	bot.send_photo(chat_id=chat_id, photo=open(board_img_path, 'rb'))

	positions_im, tile_im = board.get_possible_positions(card_image)
	
	board_tile_path = field_image_path.replace('image', 'tile_im').replace('.png', '.jpg')
	tile_im.save(board_tile_path)
	bot.send_message(chat_id=chat_id, text='Detected tile:')
	bot.send_photo(chat_id=chat_id, photo=open(board_tile_path, 'rb'))

	board_possible_path = field_image_path.replace('image', 'positions_im').replace('.png', '.jpg')
	cv2.imwrite(board_possible_path, positions_im)
	bot.send_message(chat_id=chat_id, text='Available tile positions:')
	bot.send_photo(chat_id=chat_id, photo=open(board_possible_path, 'rb'))

	board_tiles_lefte_path = field_image_path.replace('image', 'tiles_left_im').replace('.png', '.jpg')
	tiles_left_im = board.get_tiles_left()
	tiles_left_im.save(board_tiles_lefte_path)
	bot.send_message(chat_id=chat_id, text='Tiles available:')
	bot.send_photo(chat_id=chat_id, photo=open(board_tiles_lefte_path, 'rb'))


if __name__ == "__main__":
	with open('config.json') as f:
		config = json.load(f)

	BOT_TOKEN = config['TG_BOT_KEY']

	handlers_fns = {
		'classifyGame': gameClassifier.classifyGameImage, 
		'detect_checkers': detect_checkers,
		'detect_tictactoe': detect_tictactoe,
		'detect_carcassone': detect_carcassone
	}
	bot = tgbot.initBot(token=BOT_TOKEN, handlers_fns=handlers_fns)
