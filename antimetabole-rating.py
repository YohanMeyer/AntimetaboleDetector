import os
import jsonlines
import sys
import json
from scipy.spatial import distance
import numpy as np
from tqdm import tqdm
import fasttext
from os.path import join
import pickle
from nltk import bigrams
from nltk import trigrams
import prosodic

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor

NUMBER_OF_FOLDS = 5
RANDOM_SEED = 1

class AntimetaboleRatingEngine:
    def __init__(self, fasttextModel = None, featureTypes = None, C = 1, modelType = "log_reg"):
       
        if fasttextModel is not None:
            self.fasttextModel = fasttext.load_model(fasttextModel)
        else:
            self.fasttextModel = None
            print("Warning : no embedding model.")

        self.dubNegList = ["no", "not", "never", "nothing"]
        self.dubConjList = ["and", "as", "because", "for", "yet", "nor", "so", "or", "but"]
        self.dubHardPunctList = [":", ".", ";", "*", "?", "!", ")", "(", "[", "]", "\""]
        self.dubSoftPunctList = [","]
        
        self.featureTypes = featureTypes

        self.summary = None
        self.C = C # Inverse of regularization strength

        self.modelType = modelType
        self.model = None

        self.negativeAnnotation = "NotSalient"
        self.duplicateAnnotation = "Duplicate"
    
    def get_file_jsonlines(self, fileName, folder):
        filePath = os.path.join(folder, fileName)
        jsonLines = []
        
        try:
            with jsonlines.open(filePath) as file:
                print("Loading content of", filePath, '...')
                for lineJson in file:
                    jsonLines.append(lineJson)
                file.close()
                print("\nFile loaded !")
                print("-------------\n")
                return jsonLines
        except IOError:
            print("File", filePath, "not found.")
            return -1
    
    def get_dubremetz_features(self, candidate):

        lemmaIds = candidate["lemmaIndex"]
        indexA1 = lemmaIds[0]
        indexA2 = lemmaIds[-1]
        indexB1 = lemmaIds[int(len(lemmaIds)/2 - 1)]
        indexB2 = lemmaIds[int(len(lemmaIds)/2)]

        words = candidate["words"]
        lemmas = candidate["lemmas"]
        dep = candidate["dep"]
        
        beforeA1 = words[:indexA1]
        betweenA1B1 = words[indexA1 + 1 : indexB1]
        betweenB1B2 = words[indexB1 + 1 : indexB2]
        betweenB2A2 = words[indexB2 + 1 : indexA2]

        dubConjList = self.dubConjList
        dubNegList = self.dubNegList

        features = []

        hardPunctList = self.dubHardPunctList
        softPunctList = self.dubSoftPunctList

        # Basic

        numHardPunct = 0
        numSoftPunct = 0
        for word in betweenA1B1:
            if word in hardPunctList:
                numHardPunct += 1
            if word in softPunctList:
                numSoftPunct += 1
                
        for word in betweenB2A2:
            if word in hardPunctList:
                numHardPunct += 1
            if word in softPunctList:
                numSoftPunct += 1
        # punct
        features.append(numHardPunct)
        # softPunct
        features.append(numSoftPunct)

        numHardPunct = 0
        for word in betweenB1B2:
            if word in hardPunctList:
                numHardPunct += 1
        # centralPunct
        features.append(numHardPunct)

        repetitionA1 = lemmas.count(lemmas[indexA1]) - 2
        repetitionB1 = lemmas.count(lemmas[indexB1]) - 2
        # mainRep
        features.append(repetitionA1 + repetitionB1)

        # Size

        sizeDifference = abs((indexB1 - indexA1) - (indexA2 - indexB2))
        # diffSize
        features.append(sizeDifference)

        centralDistance = indexB2 - indexB1
        # toksInBC
        features.append(centralDistance)

        # Similarity

        mergeBetweenA1B1 = " ".join(betweenA1B1)
        mergeBetweenB2A2 = " ".join(betweenB2A2)
        features.append(mergeBetweenA1B1 == mergeBetweenB2A2)

        sameTokens = 0
        for lemma in lemmas[indexA1 + 1 : indexB1]:
            if lemma in lemmas[indexB2 + 1 : indexA2]: 
                sameTokens += 1
        # sameTok
        features.append(sameTokens)

        similarityScore = sameTokens / (indexB1 - indexA1 + indexA2 - indexB2)
        # simScore
        features.append(similarityScore)

        bigrams1 = bigrams(mergeBetweenA1B1)
        bigrams2 = bigrams(mergeBetweenB2A2)
        bigramScore = len(set(bigrams1).intersection(set(bigrams2)))
        # sameBigram
        features.append(bigramScore)

        trigrams1 = trigrams(mergeBetweenA1B1)
        trigrams2 = trigrams(mergeBetweenB2A2)
        trigramScore = len(set(trigrams1).intersection(set(trigrams2)))
        # sameTrigram
        features.append(trigramScore)

        
        simBeginning = len(set(beforeA1).intersection(set(betweenB1B2)))
        # sameCont
        features.append(simBeginning)

        # Lexical clues

        hasConj = 0
        for word in betweenB1B2:
            if word in dubConjList:
                hasConj = 1
        # hasConj
        features.append(hasConj)

        hasNeg = 0
        if len(set(dubNegList).intersection(set(words))) >= 1:
            hasNeg = 1
        # hasNeg
        features.append(hasNeg)

        hasTo = 0
        if (("to" in betweenA1B1 and "to" in betweenB2A2)
                or ("into" in betweenA1B1 and "into" in betweenB2A2)
                or ("from" in words and "to" in words and words.index("from") < words.index("to"))):
            hasTo = 1
        # hasTo
        features.append(hasTo)


        # Syntactic Features
        
        # sameTag
        if dep[indexA1] == dep[indexB1] == dep[indexB2] == dep[indexA2]:
            features.append(1)
        else:
            features.append(0)
            
        # sameDepWaWb'
        if dep[indexA1] == dep[indexB2]:
            features.append(1)
        else:
            features.append(0)
        
        # sameDepWaWa'
        if dep[indexA1] == dep[indexA2]:
            features.append(1)
        else:
            features.append(0)
        
        # sameDepWbWb'
        if dep[indexB1] == dep[indexB2]:
            features.append(1)
        else:
            features.append(0)
        
        # sameDepWbWa'
        if dep[indexB1] == dep[indexA2]:
            features.append(1)
        else:
            features.append(0)

        return features
    
    def get_nested_dubremetz_features(self, candidate):

        lemmaIds = candidate["lemmaIndex"]
        nbTerms = len(lemmaIds)
        nbIntervals = nbTerms / 2 - 1
        words = candidate["words"]
        lemmas = candidate["lemmas"]
        dep = candidate["dep"]
        
        middleTermLeftId = lemmaIds[int(nbTerms / 2) - 1]
        middleTermRightId = lemmaIds[int(nbTerms / 2)]
        
        beforeA1 = words[:lemmaIds[0]]
        wordsBetweenMiddleTerms = words[middleTermLeftId + 1 : middleTermRightId]
        
        wordsBetweenTermsLeft = []
        wordsBetweenTermsRight = []
        lemmasBetweenTermsLeft = []
        lemmasBetweenTermsRight = []
    
        for index in range(int(nbTerms / 2) - 1):
            wordsBetweenTermsLeft.append(
                    words[lemmaIds[index] + 1 : lemmaIds[index + 1]]
            )
            wordsBetweenTermsRight.append(
                    words[lemmaIds[index + int(nbTerms / 2)] + 1 : lemmaIds[index + int(nbTerms / 2) + 1]]
            )
            lemmasBetweenTermsLeft.append(
                    lemmas[lemmaIds[index] + 1 : lemmaIds[index + 1]]
            )
            lemmasBetweenTermsRight.append(
                    lemmas[lemmaIds[index + int(nbTerms / 2)] + 1 : lemmaIds[index + int(nbTerms / 2) + 1]]
            )
        
        dubConjList = self.dubConjList
        dubNegList = self.dubNegList

        features = []

        hardPunctList = self.dubHardPunctList
        softPunctList = self.dubSoftPunctList

        # Basic

        numHardPunct = 0
        numSoftPunct = 0
        
        for interval in wordsBetweenTermsLeft:
            for word in interval:
                if word in hardPunctList:
                    numHardPunct += 1
                if word in softPunctList:
                    numSoftPunct += 1
                
        for interval in wordsBetweenTermsRight:
            for word in interval:
                if word in hardPunctList:
                    numHardPunct += 1
                if word in softPunctList:
                    numSoftPunct += 1
        
        # punct
        features.append(numHardPunct / nbIntervals)
        # softPunct
        features.append(numSoftPunct / nbIntervals)

        numHardPunct = 0
        for word in wordsBetweenMiddleTerms:
            if word in hardPunctList:
                numHardPunct += 1
        # centralPunct
        features.append(numHardPunct)

        repetition = 0
        for index in range(int(nbTerms / 2)):
            repetition += lemmas.count(lemmas[lemmaIds[index]]) - 2
            
        # mainRep
        features.append(repetition)

        # Size

        sizeDifference = 0
        for leftInterval, rightInterval in zip(wordsBetweenTermsLeft, wordsBetweenTermsRight):
            sizeDifference += abs(len(leftInterval) - len(rightInterval))
        
        # diffSize
        features.append(sizeDifference)

        centralDistance = len(wordsBetweenMiddleTerms)
        # toksInBC
        features.append(centralDistance)

        # Similarity

        similarIntervals = 1
        for leftInterval, rightInterval in zip(wordsBetweenTermsLeft, wordsBetweenTermsRight):
            if leftInterval != rightInterval:
                similarIntervals = 0
        # exactMatch
        features.append(similarIntervals)

        sameTokens = 0
        nbTokens = 0
        for leftInterval, rightInterval in zip(lemmasBetweenTermsLeft, lemmasBetweenTermsRight):
            nbTokens += (2 + len(leftInterval) + len(rightInterval))
            for lemma in leftInterval:
                if lemma in rightInterval:
                    sameTokens += 1
        # sameTok
        features.append(sameTokens)

        similarityScore = sameTokens / nbTokens
        # simScore
        features.append(similarityScore)

        bigramScore = 0
        trigramScore = 0
        for leftInterval, rightInterval in zip(wordsBetweenTermsLeft, wordsBetweenTermsRight):
            bigrams1 = bigrams(" ".join(leftInterval))
            bigrams2 = bigrams(" ".join(rightInterval))
            bigramScore += len(set(bigrams1).intersection(set(bigrams2)))
            
            trigrams1 = trigrams(" ".join(leftInterval))
            trigrams2 = trigrams(" ".join(rightInterval))
            trigramScore += len(set(trigrams1).intersection(set(trigrams2)))
            
        # sameBigram
        features.append(bigramScore)
        # sameTrigram
        features.append(trigramScore)
        
        simBeginning = len(set(beforeA1).intersection(set(wordsBetweenMiddleTerms)))
        # sameCont
        features.append(simBeginning)

        # Lexical clues

        hasConj = 0
        for word in wordsBetweenMiddleTerms:
            if word in dubConjList:
                hasConj = 1
        # hasConj
        features.append(hasConj)

        hasNeg = 0
        if len(set(dubNegList).intersection(set(words))) >= 1:
            hasNeg = 1
        # hasNeg
        features.append(hasNeg)

        hasTo = 0
        for leftInterval, rightInterval in zip(wordsBetweenTermsLeft, wordsBetweenTermsRight):
            if (("to" in leftInterval and "to" in rightInterval)
                    or ("into" in leftInterval and "into" in rightInterval)
                    or ("from" in words and "to" in words and words.index("from") < words.index("to"))):
                hasTo = 1
        # hasTo
        features.append(hasTo)


        # Syntactic Features
        
        # sameTag
        depTerms = [dep[termId] for termId in lemmaIds]
        if depTerms.count(depTerms[0]) == len(depTerms):
            features.append(1)
        else:
            features.append(0)
        
        sameDepCount = 0
        nbDep = 0
        
        for index1 in range(int(nbTerms / 2)):
            for index2 in range(int(nbTerms / 2), nbTerms):
                sameDepCount += int(depTerms[index1] == depTerms[index2])
                nbDep += 1
        
        # SameDeps
        features.append(sameDepCount / nbDep)
        
        return features
    
    def get_improved_dubremetz_features(self, candidate):

        lemmaIds = candidate["lemmaIndex"]
        indexA1 = lemmaIds[0]
        indexA2 = lemmaIds[-1]
        indexB1 = lemmaIds[int(len(lemmaIds)/2 - 1)]
        indexB2 = lemmaIds[int(len(lemmaIds)/2)]

        words = candidate["words"]
        lemmas = candidate["lemmas"]
        dep = candidate["dep"]
        
        beforeA1 = words[:indexA1]
        betweenA1B1 = words[indexA1 + 1 : indexB1]
        betweenB1B2 = words[indexB1 + 1 : indexB2]
        betweenB2A2 = words[indexB2 + 1 : indexA2]

        dubConjList = self.dubConjList
        dubConjList += ["although", "before", "once", "though", "while", "and"]
        dubNegList = self.dubNegList
        dubNegList += ["neither", "none"]

        features = []
        
        hardPunctList = self.dubHardPunctList
        hardPunctList += ["-", "–", "{", "}", "'", "\""]
        softPunctList = self.dubSoftPunctList

        # Basic

        numHardPunct = 0
        numSoftPunct = 0
        for word in betweenA1B1:
            if word in hardPunctList:
                numHardPunct += 1
            if word in softPunctList:
                numSoftPunct += 1
                
        for word in betweenB2A2:
            if word in hardPunctList:
                numHardPunct += 1
            if word in softPunctList:
                numSoftPunct += 1
        # punct
        features.append(numHardPunct)
        # softPunct
        features.append(numSoftPunct)

        numHardPunct = 0
        numSoftPunct = 0
        for word in betweenB1B2:
            if word in hardPunctList:
                numHardPunct += 1
            if word in softPunctList:
                numSoftPunct += 1
        # centralPunct
        features.append(numHardPunct)
        features.append(numSoftPunct)

        repetitionA1 = lemmas.count(lemmas[indexA1]) - 2
        repetitionB1 = lemmas.count(lemmas[indexB1]) - 2
        # mainRep
        features.append(repetitionA1 + repetitionB1)

        # Size

        sizeDifference = abs((indexB1 - indexA1) - (indexA2 - indexB2))
        # diffSize
        features.append(sizeDifference)

        centralDistance = indexB2 - indexB1
        # toksInBC
        features.append(centralDistance)

        # Similarity

        mergeBetweenA1B1 = " ".join(betweenA1B1)
        mergeBetweenB2A2 = " ".join(betweenB2A2)
        lemmasMergedBetweenA1B1 = " ".join(lemmas[indexA1 + 1 : indexB1])
        lemmasMergedBetweenB2A2 = " ".join(lemmas[indexB2 + 1 : indexA2])
        features.append(lemmasMergedBetweenA1B1 == lemmasMergedBetweenB2A2)

        sameTokens = 0
        for lemma in lemmas[indexA1 + 1 : indexB1]:
            if lemma in lemmas[indexB2 + 1 : indexA2]: 
                sameTokens += 1
        # sameTok
        features.append(sameTokens)

        similarityScore = sameTokens / (indexB1 - indexA1 + indexA2 - indexB2)
        # simScore
        features.append(similarityScore)

        bigrams1 = bigrams(mergeBetweenA1B1)
        bigrams2 = bigrams(mergeBetweenB2A2)
        bigramScore = len(set(bigrams1).intersection(set(bigrams2)))
        # sameBigram
        features.append(bigramScore)

        trigrams1 = trigrams(mergeBetweenA1B1)
        trigrams2 = trigrams(mergeBetweenB2A2)
        trigramScore = len(set(trigrams1).intersection(set(trigrams2)))
        # sameTrigram
        features.append(trigramScore)

        
        simBeginning = len(set(beforeA1).intersection(set(betweenB1B2)))
        # sameCont
        features.append(simBeginning)

        # Lexical clues

        hasConjMiddle = 0
        for word in betweenB1B2:
            if word in dubConjList:
                hasConjMiddle = 1
        # hasConj
        features.append(hasConjMiddle)
        
        hasConjLeft = 0
        for word in betweenA1B1:
            if word in dubConjList:
                hasConjLeft = 1
        features.append(hasConjLeft)
        
        hasConjRight = 0
        for word in betweenB2A2:
            if word in dubConjList:
                hasConjRight = 1
        features.append(hasConjRight)

        hasNeg = 0
        if len(set(dubNegList).intersection(set(words))) >= 1:
            hasNeg = 1
        # hasNeg
        features.append(hasNeg)

        hasTo = 0
        if (("to" in betweenA1B1 and "to" in betweenB2A2)
                or ("into" in betweenA1B1 and "into" in betweenB2A2)
                or ("from" in words and "to" in words and words.index("from") < words.index("to"))):
            hasTo = 1
        # hasTo
        features.append(hasTo)


        # Syntactic Features
        
        # sameDepWaWb'
        if dep[indexA1] == dep[indexB2]:
            features.append(1)
        else:
            features.append(0)
        
        # sameDepWaWa'
        if dep[indexA1] == dep[indexA2]:
            features.append(1)
        else:
            features.append(0)
        
        # sameDepWbWb'
        if dep[indexB1] == dep[indexB2]:
            features.append(1)
        else:
            features.append(0)
        
        # sameDepWbWa'
        if dep[indexB1] == dep[indexA2]:
            features.append(1)
        else:
            features.append(0)
            
        isNested = (len(lemmaIds) > 4)
        features.append(isNested)

        return features
    
    def get_schneider_features(self, candidate):

        # embedding features
        
        lemmaIds = candidate["lemmaIndex"]
        
        indexA1 = lemmaIds[0]
        indexA2 = lemmaIds[-1]
        indexB1 = lemmaIds[int(len(lemmaIds)/2 - 1)]
        indexB2 = lemmaIds[int(len(lemmaIds)/2)]
        simplifiedTermsIds = [indexA1, indexA2, indexB1, indexB2]
        
        vectors = candidate["vectors"]
        lemmas = candidate["lemmas"]
        features = []

        for index1 in simplifiedTermsIds:
            if vectors[index1] is not None and len(vectors[index1]) == 0:
                print("Problem with embedding vector (",  index1, ").")
                return
            for index2 in simplifiedTermsIds:
                if index2 <= index1:
                    continue
                if vectors[index1] is None or vectors[index2] is None:
                    features.append(1)
                else:
                    features.append(distance.cosine(vectors[index1], vectors[index2]))
        
        # lexical features
        
        nbTerms = len(simplifiedTermsIds)
        
        for index1 in range(nbTerms):
            for index2 in range(index1 + 1, nbTerms):
                features.append(int(
                        lemmas[simplifiedTermsIds[index1]] == 
                        lemmas[simplifiedTermsIds[index2]]))
        
        return np.asarray(features)
    
    def get_nested_schneider_features(self, candidate):

        # embedding features
        
        lemmaIds = candidate["lemmaIndex"]
        vectors = candidate["vectors"]
        lemmas = candidate["lemmas"]
        features = []

        totalDistance = 0.0
        nbLemmas = 0.0
        for index1 in lemmaIds:
            if vectors[index1] is not None and len(vectors[index1]) == 0:
                print("Problem with embedding vector (",  index1, ").")
                return
            for index2 in lemmaIds:
                if index2 <= index1:
                    continue
                if vectors[index1] is not None and vectors[index2] is not None:
                    totalDistance += distance.cosine(vectors[index1], vectors[index2])
                    nbLemmas += 1.0
        features.append(totalDistance / nbLemmas)
        
        # lexical features
        
        nbTerms = len(lemmaIds)    
        nbIdentical = 0
        nbPairs = 0
        
        for index1 in range(nbTerms):
            for index2 in range(index1 + 1, nbTerms):
                nbIdentical += int(
                        lemmas[lemmaIds[index1]] == 
                        lemmas[lemmaIds[index2]])
                nbPairs += 1
        features.append(nbIdentical / nbPairs)
        
        return np.asarray(features)
    
    def get_improved_schneider_features(self, candidate):

        # embedding features
        # improvement : only consider the matching pairs
        
        lemmaIds = candidate["lemmaIndex"]
        nbTerms = len(lemmaIds)
        vectors = candidate["vectors"]
        lemmas = candidate["lemmas"]
        
        features = []
        totalDistance = 0.0
        nbLemmas = 0

        for index in range(int(nbTerms / 2)):
            lemmaId1 = index
            lemmaId2 = nbTerms - index - 1
            if vectors[lemmaId1] is not None and len(vectors[lemmaId1]) == 0:
                print("Problem with embedding vector (",  lemmaId1, ").")
                return
            elif vectors[lemmaId1] is not None and vectors[lemmaId2] is not None:
                totalDistance += distance.cosine(vectors[lemmaId1], vectors[lemmaId2])
                nbLemmas += 1
                
        features.append(totalDistance / nbLemmas)
        
        # lexical features
        # improvement : only consider the matching pairs
        
        nbIdentical = 0
        
        for index in range(int(nbTerms / 2)):
            lemmaId1 = index
            lemmaId2 = nbTerms - index - 1
            nbIdentical += int(lemmas[lemmaId1] == lemmas[lemmaId2])
            
        features.append(nbIdentical / (nbTerms / 2))
            
        isNested = (len(lemmaIds) > 4)
        features.append(isNested)
        
        return np.asarray(features)
    
    def get_parison_features(self, candidate):

        lemmaIds = candidate["lemmaIndex"]
        nbTerms = len(lemmaIds)
        nbIntervals = nbTerms / 2 - 1
        dep = candidate["dep"]
        
        middleTermLeftId = lemmaIds[int(nbTerms / 2) - 1]
        middleTermRightId = lemmaIds[int(nbTerms / 2)]
        
        beforeA1 = dep[ : lemmaIds[0]]
        afterA2 = dep[lemmaIds[-1] + 1 : ]
        
        depBetweenMiddleTerms = dep[middleTermLeftId + 1 : middleTermRightId]
        depBetweenTermsLeft = []
        depBetweenTermsRight = []
    
        for index in range(int(nbTerms / 2) - 1):
            depBetweenTermsLeft.append(
                    dep[lemmaIds[index] + 1 : lemmaIds[index + 1]]
            )
            depBetweenTermsRight.append(
                    dep[lemmaIds[index + int(nbTerms / 2)] + 1 : lemmaIds[index + int(nbTerms / 2) + 1]]
            )
            
        features = []
        correspondingPos = 0
        
        for posIntervalLeft, posIntervalRight in zip(depBetweenTermsLeft, depBetweenTermsRight):
            for posTagLeft, posTagRight in zip(posIntervalLeft, posIntervalRight):
                if posTagLeft == posTagRight:
                    correspondingPos += 1
                else:
                    break
        # parison between terms
        features.append(correspondingPos / nbIntervals)
        
        correspondingPos = 0
        for posTagLeft, posTagRight in zip(beforeA1, depBetweenMiddleTerms):
            if posTagLeft == posTagRight:
                correspondingPos += 1
            else:
                break
        # parison in introduction
        features.append(correspondingPos / nbIntervals)
        
        correspondingPos = 0
        for posTagLeft, posTagRight in zip(depBetweenMiddleTerms, afterA2):
            if posTagLeft == posTagRight:
                correspondingPos += 1
            else:
                break
        # parison in conclusion
        features.append(correspondingPos / nbIntervals)
        
        return features
    
    def get_isocolon_features(self, candidate):

        lemmaIds = candidate["lemmaIndex"]
        nbTerms = len(lemmaIds)
        nbIntervals = nbTerms / 2 - 1
        words = candidate["words"]
        
        middleTermLeftId = lemmaIds[int(nbTerms / 2) - 1]
        middleTermRightId = lemmaIds[int(nbTerms / 2)]
        
        beforeA1 = words[ : lemmaIds[0]]
        afterA2 = words[lemmaIds[-1] + 1 : ]
        
        wordsBetweenMiddleTerms = words[middleTermLeftId + 1 : middleTermRightId]
        wordsBetweenTermsLeft = []
        wordsBetweenTermsRight = []
    
        for index in range(int(nbTerms / 2) - 1):
            wordsBetweenTermsLeft.append(
                    words[lemmaIds[index] + 1 : lemmaIds[index + 1]]
            )
            wordsBetweenTermsRight.append(
                    words[lemmaIds[index + int(nbTerms / 2)] + 1 : lemmaIds[index + int(nbTerms / 2) + 1]]
            )
            
        features = []
        
        #isocolon in terms
        nbSyllTerms = [len(prosodic.Word(words[termId]).syllables()) for termId in lemmaIds]
        # 1 if all terms have the same number of syllables, else 0
        features.append(int(len(set(nbSyllTerms)) == 1))
        
        # isocolon in intervals between terms
        correspondingIntervals = 0
        
        for intervalLeft, intervalRight in zip(wordsBetweenTermsLeft, wordsBetweenTermsRight):
            # counts the total number of syllables in an interval between two terms
            nbSyllablesLeft = sum([len(prosodic.Word(word).syllables()) for word in intervalLeft])
            nbSyllablesRight = sum([len(prosodic.Word(word).syllables()) for word in intervalRight])
            
            if nbSyllablesLeft == nbSyllablesRight:
                correspondingIntervals += 1
        
        features.append(correspondingIntervals / nbIntervals)
        
        # isocolon in introduction
        nbSyllablesBeforeA1 = sum([len(prosodic.Word(word).syllables()) for word in beforeA1])
        nbSyllablesMiddle = sum([len(prosodic.Word(word).syllables()) for word in wordsBetweenMiddleTerms])
            
        features.append(int(nbSyllablesBeforeA1 == nbSyllablesMiddle))
        
        # isocolon in conclusion
        nbSyllablesMiddle = sum([len(prosodic.Word(word).syllables()) for word in wordsBetweenMiddleTerms])
        nbSyllablesAfterA2 = sum([len(prosodic.Word(word).syllables()) for word in afterA2])
        
        features.append(int(nbSyllablesMiddle == nbSyllablesAfterA2))
        
        return features
    
    def get_nominal_groups_features(self, candidate):

        lemmaIds = candidate["lemmaIndex"]
        lemmas = candidate["lemmas"]
        nbTerms = len(lemmaIds)
        nbWords = len(lemmas)
        nbIntervals = nbTerms / 2 - 1
            
        features = []
        
        totalNbWords = 0
        for index in range(int(nbTerms / 2)):
            termId = lemmaIds[index]
            
            
            matchingIndex = nbTerms - index - 1
            matchingTermId = lemmaIds[matchingIndex]
            
            previousTermId = -1
            nextMatchingTermId = -1
            if(index > 0):
                previousTermId = lemmaIds[index - 1]
                nextMatchingTermId = lemmaIds[index + 1]

            nextTermId = lemmaIds[index + 1]
            previousMatchingTermId = lemmaIds[index - 1]
            
            # check backwards
            offset = 0
            while True:
                offset -= 1
                if ((termId + offset) < 0
                        or (termId + offset) == previousTermId
                        or (matchingTermId + offset) == previousMatchingTermId):
                    break
                elif lemmas[termId + offset] == lemmas[matchingTermId + offset]:
                    totalNbWords += 1
                else:
                    break
            
            # check forwards
            offset = 0
            while True:
                offset += 1
                if ((matchingTermId + offset) >= nbWords
                        or (termId + offset) == nextTermId
                        or (matchingTermId + offset) == nextMatchingTermId):
                    break
                elif lemmas[termId + offset] == lemmas[matchingTermId + offset]:
                    totalNbWords += 1
                else:
                    break
        
        features.append(totalNbWords / (nbTerms / 2))
        
        return features
    
    def get_repetitions_features(self, candidate):
        
        lemmaIds = candidate["lemmaIndex"]
        lemmas = candidate["lemmas"]
        nbTerms = len(lemmaIds)
        nbWords = len(lemmas)
        
        middleTermLeftId = lemmaIds[int(nbTerms / 2) - 1]
        middleTermRightId = lemmaIds[int(nbTerms / 2)]
        firstTermId = lemmaIds[0]
        lastTermId = lemmaIds[-1]
        
        features = []
        
        # check backwards
        totalRepeatedWords = 0
        offset = 0
        
        while True:
            offset -= 1
            if (firstTermId + offset) < 0 or (middleTermRightId + offset) == middleTermLeftId:
                break
            elif lemmas[firstTermId + offset] == lemmas[middleTermRightId + offset]:
                totalRepeatedWords += 1
            else:
                break
        
        features.append(totalRepeatedWords)
        
        # check forwards
        totalRepeatedWords = 0
        offset = 0
        
        while True:
            offset += 1
            if (lastTermId + offset) >= nbWords or (middleTermLeftId + offset) == middleTermRightId:
                break
            elif lemmas[middleTermLeftId + offset] == lemmas[lastTermId + offset]:
                totalRepeatedWords += 1
            else:
                break
            
        features.append(totalRepeatedWords)
        
        return features
    
    def get_features(self, candidate):
        
        if self.featureTypes is None:
            print("No feature types given.")
            return
        
        possibleFeatures = {
                "all-features": [self.get_dubremetz_features, self.get_nested_dubremetz_features, self.get_improved_dubremetz_features, self.get_schneider_features, self.get_nested_schneider_features, self.get_improved_schneider_features, self.get_parison_features, self.get_isocolon_features, self.get_nominal_groups_features, self.get_repetitions_features],
                "dubremetz": self.get_dubremetz_features,
                "nested-dubremetz": self.get_nested_dubremetz_features,
                "improved-dubremetz": self.get_improved_dubremetz_features,
                "schneider": self.get_schneider_features,
                "nested-schneider": self.get_nested_schneider_features,
                "improved-schneider": self.get_improved_schneider_features,
                "parison": self.get_parison_features,
                "isocolon": self.get_isocolon_features,
                "nominal-groups": self.get_nominal_groups_features,
                "repetitions": self.get_repetitions_features
                }
        if(self.featureTypes[0] == "all-features"):
            calledFeatures = possibleFeatures["all-features"]
        else:
            calledFeatures = [possibleFeatures[feature] for feature in self.featureTypes]
            
        features = [featureFunction(candidate) for featureFunction in calledFeatures]
        return np.concatenate(features, axis=0)
    
    def get_top(self, outputFile, topNumber, ratedFile, folder, removeVectors = True):
        
        candidates = self.get_file_jsonlines(ratedFile, folder)
        if candidates == -1:
            print("Top rating aborted.")
            return
        
        ratings = [candidate["rating"] for candidate in candidates]
        sorting = np.argsort(ratings)[::-1]
        numCandidates = min(topNumber, len(candidates))
        
        with open(os.path.join(folder, outputFile), "w") as fileOut:
            for i in range(numCandidates):
                if(removeVectors and "vectors" in candidates[sorting[i]]):
                    del candidates[sorting[i]]["vectors"]                    
                fileOut.write(json.dumps(candidates[sorting[i]]))
                fileOut.write("\n")
    
    def evaluate_model(self, topNumber, testFeatures, testLabels):
        if self.model is None:
            print("No model found, evaluating aborted.")
            return
        
        ratings = self._model_prediction(self.model, self.modelType, testFeatures)
    
        numPositives = testLabels.count(1)
        
        sortedIds = ratings.argsort()
        
        
        testLabels = np.asarray(testLabels)
        sortedRatings = ratings[sortedIds[::-1]]
        testLabels = testLabels[sortedIds[::-1]]
        
        
        numRatings = min(topNumber, len(ratings))
        sortedRatings = sortedRatings[:numRatings]
        testLabels = testLabels[:numRatings]
        
        recall = testLabels.tolist().count(1)/numPositives

        averagePrecision = label_ranking_average_precision_score(
                np.asarray([testLabels]),
                np.asarray([sortedRatings])
        )
        
        return averagePrecision, recall
    
    def _save_model(self, outputFile = "rating-model.pkl", outputFolder = "models"):
        pickle.dump(self.model, open(os.path.join(outputFolder, outputFile), 'wb'))

    def _load_model(self, inputFile = "rating-model.pkl", inputFolder = "models"):
        
        filePath = os.path.join(inputFolder, inputFile)
        print("Loading model stored in", filePath, "...")
        try:
            loadedModel = pickle.load(open(filePath, 'rb'))
            self.model = loadedModel
        except IOError:
            print("Model stored in", filePath, "not found.")
            return False
        
        print("Loading model done.")
        return True
    
    def _preprocess_data(self, data):
        
        if self.fasttextModel is not None:
            for instance in data:
                if "vectors" in instance:
                    continue
                words = instance["words"]
                vectors = [self.fasttextModel[word] for word in words]
                instance["vectors"] = vectors
        else:
            print("Warning : no embedding model found.")
        
        dataFeatures = []
        dataLabels = []
        for instance in tqdm(data):
            dataFeatures.append(self.get_features(instance))
            
            # "cats" is currently imposed by the usage of Doccano
            if ("cats" in instance and self.negativeAnnotation in instance["cats"]):
                dataLabels.append(0)
            elif "cats" in instance:
                dataLabels.append(1)

        dataFeatures = np.asarray(dataFeatures)
        dataLabels = np.asarray(dataLabels)

        return dataFeatures, dataLabels

    def _model_prediction(self, model, modelType, dataFeatures):
        
        if modelType == "log_reg" or modelType == "svm_rbf":
            return model.decision_function(dataFeatures)
        elif modelType == "reg_tree" or modelType == "random_forest":
            return model.predict(dataFeatures)
        else:
            print("Model type",  modelType, "not found, prediction aborted.")
            return 0
    
    def rate_candidates(self, fileName, folder):
        
        candidates = self.get_file_jsonlines(fileName, folder)
        if candidates == -1:
            print("No candidates found, rating aborted.")
            return

        if self.model is None:
            print("No model found, rating aborted.")
            return
        
        print("\tGetting features...")
        features, _ = self._preprocess_data(candidates)

        print("\tRating candidates...")
        ratings = self._model_prediction(self.model, self.modelType, features)

        with open(os.path.join(folder, fileName), "w") as fileOut:
            for index, candidate in enumerate(tqdm(candidates)):
                
                if "vectors" in candidate and isinstance(candidate["vectors"][0], np.ndarray):
                    # from np.array to list for json.dumps
                    candidate["vectors"] = [vec.tolist() for vec in candidate["vectors"]]
                
                candidate["rating"] = ratings[index].item()
                
                fileOut.write(json.dumps(candidate))
                fileOut.write("\n")

        print("\tRating finished.")
    
    def _train(self, dataFeatures, dataLabels):
        
        model = None
        if self.modelType == "log_reg":
            # default model type
            model = make_pipeline(
                    StandardScaler(), # normalizing data
                    LogisticRegression(
                        penalty = "l2", # non-sparse weights
                        class_weight = None,
                        solver = "liblinear", # better for small datasets
                        max_iter = 500,
                        C = self.C
                    ))
        elif self.modelType.lower() == "svm_rbf":
            model = make_pipeline(
                    StandardScaler(),
                    SVC(
                        kernel = "rbf",
                        class_weight = None,
                        gamma = "scale",
                        max_iter = 1000,
                        C = self.C
                    ))
        elif self.modelType.lower() == "reg_tree":
            model = make_pipeline(
                    StandardScaler(),
                    DecisionTreeRegressor(
                        criterion = "absolute_error", 
                        # better than squared_error
                        splitter = "best",
                        max_depth = 8
                    ))
        elif self.modelType.lower() == "random_forest":
            model = make_pipeline(
                    StandardScaler(),
                    RandomForestRegressor(
                        n_estimators = 100,
                        criterion = "absolute_error", 
                        # better than squared_error
                        max_depth = 8
                    ))
        else:
            print("Training aborted, model type", self.modelType, "does not exist.")
            return
            
        model.fit(dataFeatures, dataLabels)

        scores = self._model_prediction(model, self.modelType, dataFeatures)
        trainAvPrecision = average_precision_score(dataLabels, scores, average = "macro")
        return model, trainAvPrecision

    def train(self, trainingFile, trainingFolder = "data", loadModel = True, modelFile = "rating-model.pkl", modelFolder = "models"):
        
        isLoaded = False
        if loadModel:
            isLoaded = self._load_model(modelFile, modelFolder)
        if not isLoaded:
            print("Undergoing model training...")
            data = self.get_file_jsonlines(trainingFile, trainingFolder)
            if data == -1:
                print("Training aborted.")
                return
            
            dataFeatures, dataLabels = self._preprocess_data(data)
            
            self.model, trainAvPrecision = self._train(dataFeatures, dataLabels)
            print("Model training finished.\n------------\n")
            self._save_model(modelFile, modelFolder)
    
    def pipeline_k_fold(self, trainingFile, trainingFolder = "data"):
        
        print("Undergoing model k-fold training...")
        
        data = self.get_file_jsonlines(trainingFile, trainingFolder)
        if data == -1:
            print("Training aborted.")
            return
            
        dataFeatures, dataLabels = self._preprocess_data(data)
        
        print("preprocess finished.")
        
        kFold = StratifiedKFold(
            n_splits = NUMBER_OF_FOLDS,
            shuffle = True,
            random_state = RANDOM_SEED
        )
        
        foldsRecall = []
        foldsAveragePrecisions = []

        for foldIndex, (trainIds, testIds) in enumerate(kFold.split(dataFeatures, dataLabels)):
            
            trainFeatures = [dataFeatures[id] for id in trainIds]
            trainLabels = [dataLabels[id] for id in trainIds]
            
            self.model, trainAvPrecision = self._train(trainFeatures, trainLabels)
            print("training finished.")
            
            testFeatures = [dataFeatures[id] for id in testIds]
            testLabels = [dataLabels[id] for id in testIds]            
            
            avePrecision, recall = self.evaluate_model(400, testFeatures, testLabels)
            print("evaluation finished.")
            foldsAveragePrecisions.append(avePrecision)
            foldsRecall.append(recall)
            
            print("\t", foldIndex, "th fold average precision (on 400):", avePrecision, " ; recall:", recall)
        
        averageAveragePrecision = sum(foldsAveragePrecisions) / len(foldsAveragePrecisions)
        averageRecall = sum(foldsRecall) / len(foldsRecall)
        return [averageAveragePrecision, averageRecall], trainFeatures


def main():

    # -- Initializing the project --    
    
    trainingFile = "final-dataset.jsonl"
    fileToRate = "antimetabole-candidates.jsonl"
    folder = "data"
    model = "log_reg"
    # model = "svm_rbf"
    # model = "reg_tree"
    # model = "random_forest"
    
    
    # 5-fold cross-validation
    
    # simple baseline
    print("Initializing new rating engine (logistic regression).")
    antimetRater = AntimetaboleRatingEngine(
            fasttextModel = None,
            featureTypes = ["dubremetz"],
            modelType = model
            )
    
    results, _ = antimetRater.pipeline_k_fold(
            trainingFile = trainingFile,
            trainingFolder = folder,
            )
    
    print("\n------------")
    print("Model:", model)
    print("Used features:", antimetRater.featureTypes)
    print("Average global precision and recall:", results[0], results[1])
    print("------------\n\n\n")
    
    # all features
    print("Initializing new rating engine (logistic regression).")
    antimetRater = AntimetaboleRatingEngine(
            fasttextModel = None,
            featureTypes = ["dubremetz", "parison", "isocolon", "nominal-groups", "repetitions"],
            modelType = model
            )
    
    results, _ = antimetRater.pipeline_k_fold(
            trainingFile = trainingFile,
            trainingFolder = folder,
            )
    
    print("Model:", model)
    print("Used features:", antimetRater.featureTypes)
    print("Average global precision and recall:", results[0], results[1])
    print("------------\n\n\n")
    
    
    
    # classic training
    
    # simple baseline
    print("Initializing new rating engine (logistic regression).")
    antimetRater = AntimetaboleRatingEngine(
            fasttextModel = None,
            featureTypes = ["dubremetz"],
            modelType = model
            )
    
    antimetRater.train(
            trainingFile = trainingFile,
            trainingFolder = folder,
            loadModel = True,
            modelFile = "rating-model-log-reg-baseline.pkl",
            modelFolder = "models"
            )
    
    # all features
    print("Initializing new rating engine (logistic regression).")
    antimetRater = AntimetaboleRatingEngine(
            fasttextModel = "./fasttext_models/wiki.en.bin",
            featureTypes = ["dubremetz", "schneider", "parison", "isocolon", "nominal-groups", "repetitions"],
            modelType = model
            )
    
    antimetRater.train(
            trainingFile = trainingFile,
            trainingFolder = folder,
            loadModel = False,
            modelFile = "rating-model-log-reg-all-feat.pkl",
            modelFolder = "models"
            )
    
    
    # prepare file for annotation
    antimetRater.rate_candidates(
            fileName = fileToRate,
            folder = folder)
    
    antimetRater.get_top(
            outputFile = fileToRate[:-6] + "-top500-results.jsonl", # [:-6] to remove ".jsonl"
            topNumber = 500,
            ratedFile = fileToRate,
            removeVectors = True,
            folder = folder)
    

if __name__ == "__main__":
    
    main()