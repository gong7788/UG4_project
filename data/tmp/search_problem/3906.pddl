(define (problem blocks-problem)
	(:domain blocksworld-unstack)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0 tower0)
	(:init 
		(arm-empty )
		(clear b1)
		(on-table b2)
		(clear b2)
		(on-table b3)
		(clear b3)
		(on-table b4)
		(clear b4)
		(on-table b5)
		(clear b5)
		(on-table b6)
		(clear b6)
		(on-table b7)
		(clear b7)
		(on-table b9)
		(clear b9)
		(done t0)
		(in-tower t0 tower0)
		(done tower0)
		(tower tower0)
		(on b0 t0)
		(in-tower b0 tower0)
		(done b0)
		(on b8 b0)
		(in-tower b8 tower0)
		(done b8)
		(on-table b1)
		(clear b8)
		(red b1)
		(yellow b2)
		(red b8)
		(yellow b9)
		(= (red-count tower0) 1)
		(= (blue-count tower0) 0)
		(= (green-count tower0) 0)
		(= (yellow-count tower0) 0)
		(= (purple-count tower0) 0)
		(= (orange-count tower0) 0)
		(= (pink-count tower0) 0)
	)
	(:goal (and (forall (?x) (done ?x)) (forall (?y) (or (not (yellow ?y)) (exists (?x) (and (red ?x) (on ?x ?y))))) (forall (?x) (or (not (red ?x)) (exists (?y) (and (yellow ?y) (on ?x ?y))))) (not (on b1 b8))))
)