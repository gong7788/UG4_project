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
		(clear b4)
		(on-table b7)
		(clear b7)
		(in-tower t0)
		(on b9 t0)
		(in-tower b9)
		(on b8 b9)
		(in-tower b8)
		(on b5 b8)
		(in-tower b5)
		(on b6 b5)
		(in-tower b6)
		(on b4 b6)
		(in-tower b4)
		(yellow b0)
		(red b1)
		(blue b1)
		(blue b2)
		(red b3)
		(blue b3)
		(red b4)
		(blue b4)
		(red b5)
		(yellow b6)
		(green b7)
		(blue b8)
		(red b9)
		(blue b9)
		(green b6)
		(red b7)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?y) (or (not (yellow ?y)) (exists (?x) (and (green ?x) (on ?x ?y))))) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (forall (?y) (or (not (blue ?y)) (exists (?x) (and (red ?x) (on ?x ?y))))) (and (not (on b7 b8)) (not (on b6 b8)) (not (on b7 b5)) (not (on b7 b6)) (not (on b7 b4)))))
)