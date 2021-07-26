import pickle


if(__name__=="__main__"):
    pickle_file = "test.pickle"
    img_name = "test-12219-0.png"
    with open(pickle_file,"rb") as f:
        data = pickle.load(f)
        # for i in data:
        #     print(i)
        matches = [x for x in data if(img_name in x[0])]
        print(matches)
