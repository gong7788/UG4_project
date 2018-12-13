(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(on-table b0)
		(clear b0)
		(on-table b1)
		(clear b1)
		(on-table b2)
		(clear b2)
		(on-table b3)
		(clear b3)
		(clear b5)
		(on-table b7)
		(clear b7)
		(in-tower t0)
		(on b9 t0)
		(in-tower b9)
		(on b4 b9)
		(in-tower b4)
		(on b8 b4)
		(in-tower b8)
		(on b6 b8)
		(in-tower b6)
		(on b5 b6)
		(in-tower b5)
		(green b1)
		(blue b1)
		(yellow b2)
		(yellow b5)
		(green b7)
		(red b7)
		(green b8)
		(yellow b9)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?x) (or (not (green ?x)) (exists (?y) (and (yellow ?y) (on ?x ?y))))) (forall (?y) (or (not (yellow ?y)) (exists (?x) (and (green ?x) (on ?x ?y))))) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (and (not (on b8 b9)) (not (on b7 b9)) (not (on b6 b9)) (not (on b5 b9)) (not (on b7 b8)) (not (on b7 b6)) (not (on b7 b5)))))
)