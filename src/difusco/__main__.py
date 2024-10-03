from difusco_edward_sun.difusco.train import arg_parser
from difusco_edward_sun.difusco.train import main as difusco_main

if __name__ == "__main__":
    args = arg_parser()
    difusco_main(args)
