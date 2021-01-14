from corpus import corpus
import time


def main() -> None:
    '''
    Main function for bot.
    '''

    current_time: float = time.time()

    c = corpus()
    c.get_all_text_data(all_in_one=True)

    duration: float = time.time() - current_time
    print(f'Reading dataset done: {duration:.2f} s')
    current_time += duration


if __name__ == '__main__':
    main()
